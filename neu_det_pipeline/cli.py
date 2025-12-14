from __future__ import annotations

import json
import logging
import sys
import subprocess
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

import typer
import torch
import shutil
from rich.console import Console

from .config import (
    ConfigBundle,
    DatasetConfig,
    GenerationConfig,
    GuidanceConfig,
    LoRAConfig,
    TextualInversionConfig,
    load_config_bundle,
)
from .data import collect_dataset, split_dataset, compute_class_counts
from .guidance import GuidanceExtractor, load_guidance_map
from .caption import CaptionGenerator, load_captions_from_file
from .textual_inversion import TextualInversionTrainer
from .lora_train import LoRATrainer
from .generation import ControlNetGenerator
from .metrics import GenerationMetricsEvaluator, save_metrics

app = typer.Typer(add_completion=False)
console = Console()
DEFAULT_BUNDLE = load_config_bundle()


def _ensure_bundle(ctx: typer.Context) -> ConfigBundle:
    bundle = ctx.obj
    if isinstance(bundle, ConfigBundle):
        return bundle
    return DEFAULT_BUNDLE


def _load_lora_metadata(lora_path: Path, fallback_cfg: LoRAConfig) -> Dict[str, Any]:
    cfg_file = lora_path.parent / "lora_config.json"
    if cfg_file.exists():
        try:
            return json.loads(cfg_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            console.print(f"[yellow]无法解析 {cfg_file}，将使用默认 LoRA 配置。[/yellow]")
    return {
        "model_id": fallback_cfg.model_id,
        "rank": fallback_cfg.rank,
        "alpha": fallback_cfg.alpha,
        "dropout_rate": getattr(fallback_cfg, "dropout_rate", 0.0),
        "target_modules": [],
        "resolution": fallback_cfg.resolution,
        "seed": fallback_cfg.seed,
        "prompt_template": fallback_cfg.prompt_template,
        "mixed_precision": fallback_cfg.mixed_precision,
        "training_hyperparameters": {
            "learning_rate": fallback_cfg.learning_rate,
            "steps": fallback_cfg.steps,
            "batch_size": fallback_cfg.batch_size,
            "gradient_accumulation_steps": fallback_cfg.gradient_accumulation_steps,
            "optimizer": "AdamW",
            "lr_scheduler": fallback_cfg.lr_scheduler,
            "lr_warmup_steps": fallback_cfg.lr_warmup_steps,
            "max_grad_norm": fallback_cfg.max_grad_norm,
        },
    }


def _build_controlnet_parameters(cfg: GenerationConfig) -> List[Dict[str, Any]]:
    params: List[Dict[str, Any]] = []
    total = len(cfg.controlnet_modalities)
    for idx, modality in enumerate(cfg.controlnet_modalities):
        model_name = cfg.controlnet_models[idx] if idx < len(cfg.controlnet_models) else None
        scale = cfg.control_scales[idx] if idx < len(cfg.control_scales) else None
        start = cfg.control_start_steps[idx] if idx < len(cfg.control_start_steps) else None
        end = cfg.control_end_steps[idx] if idx < len(cfg.control_end_steps) else None
        params.append(
            {
                "modality": modality,
                "model": model_name,
                "conditioning_scale": scale,
                "guidance_start": start,
                "guidance_end": end,
            }
        )
    # Capture any extra models without matching modality entries
    if len(cfg.controlnet_models) > total:
        for idx in range(total, len(cfg.controlnet_models)):
            params.append(
                {
                    "modality": None,
                    "model": cfg.controlnet_models[idx],
                    "conditioning_scale": cfg.control_scales[idx] if idx < len(cfg.control_scales) else None,
                    "guidance_start": cfg.control_start_steps[idx] if idx < len(cfg.control_start_steps) else None,
                    "guidance_end": cfg.control_end_steps[idx] if idx < len(cfg.control_end_steps) else None,
                }
            )
    return params


def _setup_logging(log_file: Optional[Path]) -> None:
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        log_file = log_file.expanduser().resolve()
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.insert(0, logging.FileHandler(log_file, encoding="utf-8"))
        console.print(f"日志将写入 {log_file}")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )


@app.callback()
def main(
    ctx: typer.Context,
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        help="YAML 配置文件路径（覆盖默认 config.yaml）",
        exists=True,
        dir_okay=False,
        file_okay=True,
    ),
) -> None:
    """NEU-DET data augmentation pipeline."""

    if config is not None:
        try:
            ctx.obj = load_config_bundle(config)
        except FileNotFoundError as exc:  # pragma: no cover - CLI validation
            raise typer.BadParameter(str(exc)) from exc
    else:
        ctx.obj = DEFAULT_BUNDLE


@app.command()
def prepare(
    ctx: typer.Context,
    dataset_root: Path = typer.Argument(..., exists=True, file_okay=False),
    test_size: Optional[float] = typer.Option(None, help="验证集比例 (默认取配置文件中的值)"),
) -> None:
    bundle = _ensure_bundle(ctx)
    dataset_cfg = replace(bundle.dataset, root=dataset_root)
    effective_test_size = test_size if test_size is not None else dataset_cfg.test_size
    samples = collect_dataset(dataset_cfg.root)
    splits = split_dataset(samples, test_size=effective_test_size, seed=dataset_cfg.seed)
    counts = compute_class_counts(samples)
    console.print("Loaded", len(samples), "samples")
    console.print(counts)


@app.command()
def textual_inversion(
    ctx: typer.Context,
    dataset_root: Path = typer.Argument(..., exists=True, file_okay=False),
    output_dir: Path = typer.Option(Path("outputs/textual_inversion"), file_okay=False),
) -> None:
    bundle = _ensure_bundle(ctx)
    cfg = replace(bundle.textual_inversion)
    dataset_cfg = replace(bundle.dataset, root=dataset_root)
    samples = collect_dataset(dataset_root)
    splits = split_dataset(samples, test_size=dataset_cfg.test_size, seed=dataset_cfg.seed)
    counts = compute_class_counts(samples)
    console.print("Loaded", len(samples), "samples")
    console.print(counts)
    trainer = TextualInversionTrainer(
        model_id=cfg.model_id,
        token_prefix=cfg.placeholder_prefix,
        initializer_token=cfg.initializer_token,
    )
    embeddings = trainer.train_embeddings(
        splits,
        output_dir,
        steps=cfg.steps,
        lr=cfg.learning_rate,
        batch_size=cfg.batch_size,
        resolution=cfg.resolution,
        prompt_template=cfg.prompt_template,
    )
    console.print("Created", len(embeddings), "textual inversion tokens")


@app.command()
def guidance(
    ctx: typer.Context,
    dataset_root: Path = typer.Argument(..., exists=True, file_okay=False),
    output_dir: Path = typer.Option(Path("outputs/guidance"), file_okay=False),
) -> None:
    bundle = _ensure_bundle(ctx)
    cfg = replace(bundle.guidance)
    samples = collect_dataset(dataset_root)
    extractor = GuidanceExtractor(cfg.hed_repo_id, cfg.midas_repo_id, cfg.hed_ckpt, cfg.midas_ckpt)
    outputs = extractor.batch_process(samples, output_dir)
    console.print("Stored guidance cues for", len(outputs), "samples")


@app.command()
def caption(
    ctx: typer.Context,
    dataset_root: Path = typer.Argument(..., exists=True, file_okay=False),
    output_file: Path = typer.Option(Path("outputs/captions.json"), file_okay=True),
    model_name: str = typer.Option("openai/clip-vit-large-patch14"),
    use_paper_keywords: bool = typer.Option(True, help="Use paper-specified keywords (paper format) or CLIP-selected templates"),
    lora_weight: float = typer.Option(1.0, help="LoRA weight in prompt (typically 1.0)"),
) -> None:
    """Generate automatic prompts using paper-style keyword format with LoRA weights.
    
    Paper format: "keyword1, keyword2, ..., defect-specific, loRA:token:weight"
    Example: "grayscale, greyscale, hotrolled steel strip, monochrome, no humans, 
             surface defects, texture, rolled-in scale, loRA:neudet1-v1:1"
    """
    bundle = _ensure_bundle(ctx)
    samples = collect_dataset(dataset_root)
    token_map = {cls: f"<neu_{cls}>" for cls in set(s.cls_name for s in samples)}
    generator = CaptionGenerator(model_name=model_name)
    captions = generator.generate_with_token(
        samples,
        token_map,
        output_file=output_file,
        use_paper_keywords=use_paper_keywords,
        lora_weight=lora_weight,
    )
    generator.cleanup()
    console.print(f"Generated {len(captions)} paper-style captions, saved to {output_file}")
    if use_paper_keywords:
        console.print("[cyan]Using paper-specified keyword format[/cyan]")


@app.command()
def train_lora(
    ctx: typer.Context,
    dataset_root: Path = typer.Argument(..., exists=True, file_okay=False),
    lora_dir: Path = typer.Option(Path("outputs/lora"), file_okay=False),
) -> None:
    bundle = _ensure_bundle(ctx)
    cfg = replace(bundle.lora)
    samples = collect_dataset(dataset_root)
    dataset_cfg = replace(bundle.dataset, root=dataset_root)
    splits = split_dataset(samples, test_size=dataset_cfg.test_size, seed=dataset_cfg.seed)
    trainer = LoRATrainer(cfg)
    token_map = {cls: f"<neu_{cls}>" for cls in set(s.cls_name for s in samples)}
    weights = trainer.train(splits, token_map, lora_dir)
    console.print("Saved LoRA weights to", weights)


@app.command()
def generate(
    ctx: typer.Context,
    dataset_root: Path = typer.Argument(..., exists=True, file_okay=False),
    guidance_dir: Path = typer.Argument(..., exists=True, file_okay=False),
    lora_path: Path = typer.Argument(..., exists=True),
    output_dir: Path = typer.Option(Path("outputs/generated"), file_okay=False),
    caption_file: Optional[Path] = typer.Option(None, help="Path to auto-generated captions JSON file (will auto-generate if not provided)"),
    priority_class: Optional[str] = typer.Option(None, help="Prioritize generating this defect class first (e.g., 'inclusion')"),
    max_samples: Optional[int] = typer.Option(None, help="Limit the number of samples to generate (for quick smoke tests)"),
    model_name: str = typer.Option("openai/clip-vit-large-patch14", help="CLIP model for prompt selection"),
    log_file: Optional[Path] = typer.Option(None, help="日志文件路径 (默认: 每次生成独立 run_xxx/run.log)"),
) -> None:
    """Generate images with LoRA and ControlNet using auto-generated captions."""
    bundle = _ensure_bundle(ctx)
    dataset_cfg = replace(bundle.dataset, root=dataset_root)
    samples = collect_dataset(dataset_root)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"run_{timestamp}"
    image_dir = run_dir / "images"
    artifacts_dir = run_dir / "artifacts"
    run_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    artifact_paths: Dict[str, str] = {}

    effective_log = log_file or (run_dir / "run.log")
    _setup_logging(effective_log)
    
    # Sort samples to prioritize specific class if requested
    if priority_class:
        console.print(f"Prioritizing generation for class: {priority_class}")
        samples = sorted(samples, key=lambda s: (s.cls_name != priority_class, s.image_path.stem))

    if max_samples is not None:
        samples = samples[:max_samples]
        console.print(f"Limiting generation to {len(samples)} sample(s) for this run.")
    
    cfg = replace(bundle.generation)
    generator = ControlNetGenerator(cfg)
    lora_defaults = replace(bundle.lora)
    lora_metadata = _load_lora_metadata(lora_path, lora_defaults)
    controlnet_parameters = _build_controlnet_parameters(cfg)
    lora_settings = {
        "rank": lora_metadata.get("rank"),
        "alpha": lora_metadata.get("alpha"),
        "dropout_rate": lora_metadata.get("dropout_rate"),
        "target_modules": lora_metadata.get("target_modules"),
        "model_id": lora_metadata.get("model_id"),
    }
    training_hparams = lora_metadata.get("training_hyperparameters", {})
    other_config = {
        "base_model": cfg.base_model,
        "generation_seed": cfg.seed,
        "training_resolution": lora_metadata.get("resolution"),
        "training_seed": lora_metadata.get("seed"),
        "prompt_template": lora_metadata.get("prompt_template"),
        "mixed_precision": lora_metadata.get("mixed_precision"),
    }
    token_map = {cls: f"<neu_{cls}>" for cls in set(s.cls_name for s in samples)}
    
    # Always use auto-generated captions
    if not caption_file:
        caption_file = output_dir.parent / "captions.json"
    
    if not caption_file.exists():
        console.print(f"Caption file not found. Generating captions automatically...")
        caption_generator = CaptionGenerator(model_name=model_name)
        captions = caption_generator.generate_with_token(samples, token_map, output_file=caption_file)
        caption_generator.cleanup()
        console.print(f"Generated {len(captions)} captions, saved to {caption_file}")
    else:
        console.print(f"Loading captions from {caption_file}")
        captions = load_captions_from_file(caption_file)

    # Snapshot artifacts (captions, config, LoRA metadata) for reproducibility
    if caption_file.exists():
        caption_dest = artifacts_dir / caption_file.name
        try:
            if caption_dest.resolve() != caption_file.resolve():
                shutil.copy2(caption_file, caption_dest)
        except FileNotFoundError:
            caption_dest = caption_file
        artifact_paths["captions"] = str(caption_dest.resolve())

    if bundle.source_path and bundle.source_path.exists():
        config_dest = artifacts_dir / bundle.source_path.name
        if config_dest.resolve() != bundle.source_path.resolve():
            shutil.copy2(bundle.source_path, config_dest)
        artifact_paths["config"] = str(config_dest.resolve())

    lora_config_path = lora_path.parent / "lora_config.json"
    if lora_config_path.exists():
        lora_config_dest = artifacts_dir / lora_config_path.name
        if lora_config_dest.resolve() != lora_config_path.resolve():
            shutil.copy2(lora_config_path, lora_config_dest)
        artifact_paths["lora_config"] = str(lora_config_dest.resolve())
    
    # Build prompt items using auto-generated captions
    prompt_items = [(s.image_path.stem, captions.get(s.image_path.stem, f"macro shot of {token_map[s.cls_name]} steel surface")) for s in samples]
    
    conditioning = load_guidance_map(samples, guidance_dir, cfg.controlnet_modalities)
    init_images = {s.image_path.stem: s.image_path for s in samples}
    generated_paths = generator.generate(lora_path, prompt_items, conditioning, image_dir, init_images=init_images)

    manifest: Dict[str, object]
    if generated_paths:
        metrics_dest = run_dir / "metrics.json"
        evaluator = GenerationMetricsEvaluator()
        metric_result = evaluator.evaluate(generated_paths, init_images, conditioning)
        run_details = {
            "timestamp": timestamp,
            "generated_images": len(generated_paths),
            "dataset_root": str(dataset_root.resolve()),
            "guidance_dir": str(guidance_dir.resolve()),
            "lora_path": str(lora_path.resolve()),
            "controlnet_models": cfg.controlnet_models,
            "controlnet_modalities": cfg.controlnet_modalities,
            "num_inference_steps": cfg.num_inference_steps,
            "guidance_scale": cfg.guidance_scale,
            "denoising_strength": cfg.denoising_strength,
            # Ensure JSON-serializable device field
            "device": str(evaluator.device),
            "output_directory": str(image_dir.resolve()),
            "max_samples": max_samples,
        }
        save_metrics(metric_result, metrics_dest, extra=run_details)
        artifact_paths["metrics"] = str(metrics_dest.resolve())

        manifest = {
            **run_details,
            "log_file": str(effective_log.resolve()) if effective_log else None,
            "metrics_file": str(metrics_dest.resolve()),
            "caption_file": str(caption_file.resolve()) if caption_file else None,
            "generated_files": [str(path.resolve()) for path in generated_paths],
            "lora_settings": lora_settings,
            "controlnet_parameters": controlnet_parameters,
            "training_hyperparameters": training_hparams,
            "other_config": other_config,
            "artifacts": artifact_paths,
        }
    else:
        console.print("[yellow]未生成任何图像，跳过指标评估。[/yellow]")
        manifest = {
            "timestamp": timestamp,
            "generated_images": 0,
            "dataset_root": str(dataset_root.resolve()),
            "guidance_dir": str(guidance_dir.resolve()),
            "lora_path": str(lora_path.resolve()),
            "controlnet_models": cfg.controlnet_models,
            "controlnet_modalities": cfg.controlnet_modalities,
            "num_inference_steps": cfg.num_inference_steps,
            "guidance_scale": cfg.guidance_scale,
            "denoising_strength": cfg.denoising_strength,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "output_directory": str(image_dir.resolve()),
            "max_samples": max_samples,
            "log_file": str(effective_log.resolve()) if effective_log else None,
            "metrics_file": None,
            "caption_file": str(caption_file.resolve()) if caption_file else None,
            "generated_files": [],
            "lora_settings": lora_settings,
            "controlnet_parameters": controlnet_parameters,
            "training_hyperparameters": training_hparams,
            "other_config": other_config,
            "artifacts": artifact_paths,
        }

    manifest_path = run_dir / "run_context.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    # Automatically refresh YOLO dataset with latest generated images
    try:
        console.print("[cyan]自动运行 prepare_yolo_dataset.py 以更新 YOLO 数据集...[/cyan]")
        prepare_script = Path(__file__).resolve().parent.parent / "prepare_yolo_dataset.py"
        # Place the refreshed YOLO dataset inside the current run folder
        yolo_output = run_dir / "yolo_dataset"
        cmd = [
            sys.executable,
            str(prepare_script),
            str(dataset_root),
            str(image_dir),
            "--output-dir",
            str(yolo_output),
        ]
        subprocess.run(cmd, check=True)
        console.print(f"[green]YOLO 数据集已更新: {yolo_output}[/green]")
    except Exception as exc:  # noqa: BLE001 - we want to log and continue
        console.print(f"[yellow]prepare_yolo_dataset 运行失败: {exc}。请手动检查。[/yellow]")


if __name__ == "__main__":
    app()
