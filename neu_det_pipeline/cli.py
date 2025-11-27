from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from .config import DatasetConfig, GenerationConfig, GuidanceConfig, LoRAConfig, TextualInversionConfig
from .data import collect_dataset, split_dataset, export_metadata, compute_class_counts
from .guidance import GuidanceExtractor, load_guidance_map
from .textual_inversion import TextualInversionTrainer
from .lora_train import LoRATrainer
from .generation import ControlNetGenerator

app = typer.Typer(add_completion=False)
console = Console()


@app.callback()
def main() -> None:
    """NEU-DET data augmentation pipeline."""


@app.command()
def prepare(
    dataset_root: Path = typer.Argument(..., exists=True, file_okay=False),
    out_dir: Optional[Path] = typer.Option(None, help="Optional directory to copy subset"),
    test_size: float = typer.Option(0.1),
) -> None:
    samples = collect_dataset(dataset_root)
    splits = split_dataset(samples, test_size=test_size)
    counts = compute_class_counts(samples)
    console.print("Loaded", len(samples), "samples")
    console.print(counts)
    if out_dir:
        export_metadata(splits.train, out_dir / "train_metadata.json")
        export_metadata(splits.val, out_dir / "val_metadata.json")


@app.command()
def textual_inversion(
    dataset_root: Path = typer.Argument(..., exists=True, file_okay=False),
    output_dir: Path = typer.Option(Path("outputs/textual_inversion"), file_okay=False),
) -> None:
    samples = collect_dataset(dataset_root)
    splits = split_dataset(samples)
    cfg = TextualInversionConfig()
    trainer = TextualInversionTrainer(model_id=cfg.model_id, token_prefix=cfg.placeholder_prefix)
    embeddings = trainer.train_embeddings(splits, output_dir)
    console.print("Created", len(embeddings), "textual inversion tokens")


@app.command()
def guidance(
    dataset_root: Path = typer.Argument(..., exists=True, file_okay=False),
    output_dir: Path = typer.Option(Path("outputs/guidance"), file_okay=False),
) -> None:
    samples = collect_dataset(dataset_root)
    cfg = GuidanceConfig()
    extractor = GuidanceExtractor(cfg.hed_repo_id, cfg.midas_repo_id, cfg.hed_ckpt, cfg.midas_ckpt)
    outputs = extractor.batch_process(samples, output_dir)
    console.print("Stored guidance cues for", len(outputs), "samples")


@app.command()
def train_lora(
    dataset_root: Path = typer.Argument(..., exists=True, file_okay=False),
    lora_dir: Path = typer.Option(Path("outputs/lora"), file_okay=False),
) -> None:
    samples = collect_dataset(dataset_root)
    splits = split_dataset(samples)
    cfg = LoRAConfig()
    trainer = LoRATrainer(cfg)
    token_map = {cls: f"<neu_{cls}>" for cls in set(s.cls_name for s in samples)}
    weights = trainer.train(splits, token_map, lora_dir)
    console.print("Saved LoRA weights to", weights)


@app.command()
def generate(
    dataset_root: Path = typer.Argument(..., exists=True, file_okay=False),
    guidance_dir: Path = typer.Argument(..., exists=True, file_okay=False),
    lora_path: Path = typer.Argument(..., exists=True),
    output_dir: Path = typer.Option(Path("outputs/generated"), file_okay=False),
) -> None:
    samples = collect_dataset(dataset_root)
    cfg = GenerationConfig()
    generator = ControlNetGenerator(cfg)
    token_map = {cls: f"<neu_{cls}>" for cls in set(s.cls_name for s in samples)}
    prompt_items = [(s.image_path.stem, cfg.prompt_template.format(token=token_map[s.cls_name])) for s in samples]
    conditioning = load_guidance_map(samples, guidance_dir, cfg.controlnet_modalities)
    generator.generate(lora_path, prompt_items, conditioning, output_dir)


if __name__ == "__main__":
    app()
