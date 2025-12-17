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
from .caption import CaptionGenerator, load_captions_from_file, generate_captions_with_blip2
from .textual_inversion import TextualInversionTrainer
from .lora_train import LoRATrainer
from .generation import ControlNetGenerator
from .metrics import GenerationMetricsEvaluator, save_metrics

# Import create_mixed_dataset from resplit_dataset.py
resplit_module_path = Path(__file__).resolve().parent.parent
if str(resplit_module_path) not in sys.path:
    sys.path.insert(0, str(resplit_module_path))
from resplit_dataset import create_mixed_dataset

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
    model_name: str = typer.Option("openai/clip-vit-large-patch14", help="CLIP model name (ignored if --use-blip2 is set)"),
    use_paper_keywords: bool = typer.Option(True, help="Use paper-specified keywords (paper format) or CLIP-selected templates"),
    lora_weight: float = typer.Option(1.0, help="LoRA weight in prompt (typically 1.0)"),
    use_blip2: bool = typer.Option(False, help="Use BLIP-2 for dynamic caption generation (better quality, slower)"),
    blip2_model: str = typer.Option("Salesforce/blip2-opt-2.7b", help="BLIP-2 model name (only used if --use-blip2 is set)"),
    combine_with_keywords: bool = typer.Option(True, help="Combine BLIP-2 descriptions with paper keywords (only used if --use-blip2 is set)"),
    use_clip_selection: bool = typer.Option(False, help="Use CLIP to select best template for each image (slower but more accurate, only used if --use-blip2 is False)"),
) -> None:
    """Generate automatic prompts using paper-style keyword format with LoRA weights.
    
    Paper format: "keyword1, keyword2, ..., defect-specific, loRA:token:weight"
    Example: "grayscale, greyscale, hotrolled steel strip, monochrome, no humans, 
             surface defects, texture, rolled-in scale, loRA:neudet1-v1:1"
    
    With --use-blip2: Uses BLIP-2 to generate dynamic descriptions based on image content,
    optionally combined with paper keywords for better Stable Diffusion compatibility.
    """
    bundle = _ensure_bundle(ctx)
    samples = collect_dataset(dataset_root)
    token_map = {cls: f"<neu_{cls}>" for cls in set(s.cls_name for s in samples)}
    
    if use_blip2:
        console.print(f"[cyan]Using BLIP-2 model: {blip2_model}[/cyan]")
        console.print("[yellow]BLIP-2 generation is slower but produces more detailed captions[/yellow]")
        captions = generate_captions_with_blip2(
            samples=samples,
            token_map=token_map,
            output_file=output_file,
            use_paper_keywords=use_paper_keywords,
            lora_weight=lora_weight,
            model_name=blip2_model,
            combine_with_keywords=combine_with_keywords,
        )
        console.print(f"Generated {len(captions)} BLIP-2 captions, saved to {output_file}")
        if combine_with_keywords:
            console.print("[cyan]BLIP-2 descriptions combined with paper keywords[/cyan]")
    else:
        generator = CaptionGenerator(model_name=model_name)
        captions = generator.generate_with_token(
            samples,
            token_map,
            output_file=output_file,
            use_paper_keywords=use_paper_keywords,
            lora_weight=lora_weight,
            use_clip_selection=use_clip_selection,
        )
        generator.cleanup()
        console.print(f"Generated {len(captions)} paper-style captions, saved to {output_file}")
        if use_clip_selection:
            console.print("[cyan]Using CLIP to select best template for each image[/cyan]")
        elif use_paper_keywords:
            console.print("[cyan]Using paper-specified keyword format (simple concatenation)[/cyan]")


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
    mode: str = typer.Option(
        "a2",
        help="Generation preset: a1(mask only), a2(mask+HED), a3(mask+HED+Depth)",
        case_sensitive=False,
    ),
    use_bbox_mask: bool = typer.Option(True, help="Apply soft bbox mask blending to keep background intact"),
    skip_large_bbox: bool = typer.Option(
        True,
        help="Skip samples whose bbox covers too much area (prevents full-image inpaint artifacts)",
    ),
    skip_large_bbox_ratio: float = typer.Option(
        0.6,
        help="Skip when bbox_area/(image_area) >= this ratio (suggest 0.6~0.8)",
    ),
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

    # Skip samples with overly-large bbox (inpaint would degenerate into full-image redraw)
    skipped_large_bbox_stems: List[str] = []
    if use_bbox_mask and skip_large_bbox:
        from xml.etree import ElementTree as ET

        ratio_thr = float(skip_large_bbox_ratio)
        if ratio_thr <= 0:
            ratio_thr = 0.6
        if ratio_thr > 1.0:
            ratio_thr = 1.0

        kept: List[Any] = []
        for s in samples:
            # Default NEU-DET size is 200x200; prefer reading from XML for robustness
            w = h = 200
            try:
                tree = ET.parse(s.annotation_path)
                w = int(tree.findtext("size/width") or w)
                h = int(tree.findtext("size/height") or h)
            except Exception:
                pass
            xmin, ymin, xmax, ymax = s.bbox
            bbox_area = max(0, xmax - xmin) * max(0, ymax - ymin)
            img_area = max(1, w * h)
            ratio = bbox_area / img_area
            if ratio >= ratio_thr:
                skipped_large_bbox_stems.append(s.image_path.stem)
            else:
                kept.append(s)

        if skipped_large_bbox_stems:
            console.print(
                f"[yellow]Skipping {len(skipped_large_bbox_stems)} sample(s) with large bbox "
                f"(ratio >= {ratio_thr:.2f}), e.g. {skipped_large_bbox_stems[:5]}[/yellow]"
            )
        samples = kept

    if max_samples is not None:
        samples = samples[:max_samples]
        console.print(f"Limiting generation to {len(samples)} sample(s) for this run.")
    
    cfg = replace(bundle.generation)
    mode_lc = mode.lower()
    if mode_lc == "a1":
        cfg.controlnet_models = []
        cfg.controlnet_modalities = []
        cfg.control_scales = []
        cfg.num_inference_steps = 40
        cfg.denoising_strength = 0.65
        console.print("[cyan]Mode A1: mask-only img2img (no ControlNet)[/cyan]")
    elif mode_lc == "a2":
        cfg.controlnet_models = ["lllyasviel/control_v11p_sd15_softedge"]
        cfg.controlnet_modalities = ["hed"]
        cfg.control_scales = [1.0]
        cfg.num_inference_steps = 40
        cfg.denoising_strength = 0.65
        console.print("[cyan]Mode A2: mask + HED ControlNet (scale=1.0)[/cyan]")
    elif mode_lc == "a3":
        cfg.controlnet_models = [
            "lllyasviel/control_v11p_sd15_softedge",
            "lllyasviel/control_v11f1p_sd15_depth",
        ]
        cfg.controlnet_modalities = ["hed", "depth"]
        cfg.control_scales = [1.0, 1.0]
        cfg.num_inference_steps = 40
        cfg.denoising_strength = 0.65
        console.print("[cyan]Mode A3: mask + HED+Depth ControlNet (depth=0.5)[/cyan]")
    else:
        console.print(f"[yellow]未知模式 {mode}，使用默认生成配置。[/yellow]")

    # If bbox inpaint is enabled, prefer an inpainting base model (keeps background texture much better)
    if use_bbox_mask and cfg.base_model == "runwayml/stable-diffusion-v1-5":
        cfg.base_model = "runwayml/stable-diffusion-inpainting"
        console.print("[cyan]Inpaint enabled: switching base_model -> runwayml/stable-diffusion-inpainting[/cyan]")

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
        # 默认使用 CLIP，但可以通过环境变量或配置启用 BLIP-2
        use_blip2 = False  # 可以通过参数添加
        if use_blip2:
            captions = generate_captions_with_blip2(
                samples=samples,
                token_map=token_map,
                output_file=caption_file,
                use_paper_keywords=True,
                lora_weight=1.0,
            )
        else:
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
    
    # Build prompt items using auto-generated captions.
    # Always append the textual inversion token to ensure it is actually used in the prompt.
    prompt_items = []
    for s in samples:
        token = token_map[s.cls_name]
        base_prompt = captions.get(s.image_path.stem, f"macro shot of {token} steel surface")
        prompt_items.append((s.image_path.stem, f"{base_prompt}, {token}"))
    
    if cfg.controlnet_modalities:
        conditioning = load_guidance_map(samples, guidance_dir, cfg.controlnet_modalities)
    else:
        conditioning = {}

    init_images = {s.image_path.stem: s.image_path for s in samples}
    bbox_map = {s.image_path.stem: s.bbox for s in samples}
    generated_paths = generator.generate(
        lora_path,
        prompt_items,
        conditioning,
        image_dir,
        init_images=init_images,
        bbox_map=bbox_map,
        use_bbox_mask=use_bbox_mask,
    )

    manifest: Dict[str, object]
    if generated_paths:
        metrics_dest = run_dir / "metrics.json"
        evaluator = GenerationMetricsEvaluator()
        # When bbox-mask inpaint is enabled, evaluate metrics on bbox region only
        metric_bbox_map = bbox_map if use_bbox_mask else None
        metric_result = evaluator.evaluate(
            generated_paths,
            init_images,
            conditioning,
            bbox_map=metric_bbox_map,
            bbox_source_size=(200, 200),
        )
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
            "skip_large_bbox": bool(skip_large_bbox),
            "skip_large_bbox_ratio": float(skip_large_bbox_ratio),
            "skipped_large_bbox_count": len(skipped_large_bbox_stems),
            "skipped_large_bbox_stems": skipped_large_bbox_stems[:50],
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
            "skip_large_bbox": bool(skip_large_bbox),
            "skip_large_bbox_ratio": float(skip_large_bbox_ratio),
            "skipped_large_bbox_count": len(skipped_large_bbox_stems),
            "skipped_large_bbox_stems": skipped_large_bbox_stems[:50],
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

    # Automatically create mixed dataset using resplit_dataset.py
    resplit_script = Path(__file__).resolve().parent.parent / "resplit_dataset.py"
    if resplit_script.exists():
        try:
            console.print("[cyan]自动运行 resplit_dataset.py 以创建混合数据集...[/cyan]")
            # Infer paths for resplit_dataset.py
            orig_images_dir = dataset_root / "IMAGES"
            manifest_path = output_dir.parent / "split_manifest.json"
            yolo_baseline_dir = output_dir.parent / "yolo_baseline"
            orig_labels_dir = yolo_baseline_dir / "labels" if yolo_baseline_dir.exists() else output_dir.parent / "yolo_baseline" / "labels"
            mixed_output = run_dir / "mixed_dataset"
            
            cmd = [
                sys.executable,
                str(resplit_script),
                "--orig_images_dir", str(orig_images_dir),
                "--orig_labels_dir", str(orig_labels_dir),
                "--manifest_path", str(manifest_path),
                "--new_images_dir", str(image_dir),
                "--run_output_dir", str(mixed_output),
            ]
            
            result = subprocess.run(cmd, check=False, capture_output=True, text=True, encoding="utf-8", errors="replace")
            if result.returncode == 0:
                console.print(f"[green]混合数据集已创建: {mixed_output}[/green]")
            else:
                console.print(f"[yellow]resplit_dataset.py 运行失败 (exit code {result.returncode})[/yellow]")
                if result.stderr:
                    console.print(f"[yellow]错误信息: {result.stderr[:300]}[/yellow]")
                if result.stdout:
                    console.print(f"[yellow]输出: {result.stdout[:300]}[/yellow]")
        except Exception as exc:  # noqa: BLE001 - we want to log and continue
            console.print(f"[yellow]resplit_dataset.py 运行失败: {exc}。请手动检查。[/yellow]")
    else:
        console.print(
            f"[yellow]resplit_dataset.py 未找到 ({resplit_script})，跳过混合数据集自动创建。"
            f"请手动运行 resplit_dataset.py 创建混合数据集。[/yellow]"
        )

    # Automatically create mixed dataset after image generation
    if generated_paths:
        try:
            console.print("[cyan]自动生成混合数据集...[/cyan]")
            # Try to infer paths from dataset_root and output_dir
            # First, try to find the project root (where resplit_dataset.py is located)
            project_root = Path(__file__).resolve().parent.parent
            
            # Try to find original images directory
            orig_images_dir = dataset_root / "IMAGES"
            if not orig_images_dir.exists():
                # Try project root / NEU-DET / IMAGES
                orig_images_dir = project_root / "NEU-DET" / "IMAGES"
            
            # Try to find labels directory
            orig_labels_dir = output_dir.parent / "yolo_baseline" / "labels"
            if not orig_labels_dir.exists():
                # Try project root / outputs / yolo_baseline / labels
                orig_labels_dir = project_root / "outputs" / "yolo_baseline" / "labels"
            
            # Try to find manifest file
            manifest_path = output_dir.parent / "split_manifest.json"
            if not manifest_path.exists():
                # Try project root / outputs / split_manifest.json
                manifest_path = project_root / "outputs" / "split_manifest.json"
            
            # new_images_dir is the image_dir (run_dir / "images")
            new_images_dir = image_dir
            # run_output_dir will be set to run_dir / "mixed_dataset" by default
            run_output_dir = run_dir / "mixed_dataset"
            
            # Only proceed if manifest exists
            if manifest_path.exists():
                create_mixed_dataset(
                    orig_images_dir=orig_images_dir,
                    orig_labels_dir=orig_labels_dir,
                    manifest_path=manifest_path,
                    new_images_dir=new_images_dir,
                    run_output_dir=run_output_dir,
                    seed=dataset_cfg.seed,
                )
                console.print(f"[green]混合数据集已生成: {run_output_dir}[/green]")
            else:
                console.print(f"[yellow]未找到 split_manifest.json ({manifest_path})，跳过混合数据集生成。[/yellow]")
                console.print(f"[yellow]请确保 split_manifest.json 存在于 outputs 目录下。[/yellow]")
        except Exception as exc:  # noqa: BLE001 - we want to log and continue
            console.print(f"[yellow]混合数据集生成失败: {exc}。请手动检查。[/yellow]")


if __name__ == "__main__":
    app()
