from __future__ import annotations

from pathlib import Path
from typing import Dict

from .config import DatasetConfig, GenerationConfig, GuidanceConfig, LoRAConfig, TextualInversionConfig
from .data import collect_dataset, split_dataset
from .guidance import GuidanceExtractor
from .lora_train import LoRATrainer
from .generation import ControlNetGenerator
from .textual_inversion import TextualInversionTrainer


def run_full_pipeline(
    dataset_cfg: DatasetConfig,
    ti_cfg: TextualInversionConfig,
    lora_cfg: LoRAConfig,
    guidance_cfg: GuidanceConfig,
    gen_cfg: GenerationConfig,
    work_dir: Path,
) -> None:
    work_dir.mkdir(parents=True, exist_ok=True)
    samples = collect_dataset(dataset_cfg.resolve())
    splits = split_dataset(samples, dataset_cfg.test_size, dataset_cfg.seed)

    ti_dir = work_dir / "textual_inversion"
    lora_dir = work_dir / "lora"
    guidance_dir = work_dir / "guidance"
    gen_dir = work_dir / "generated"

    textual_trainer = TextualInversionTrainer(model_id=ti_cfg.model_id, token_prefix=ti_cfg.placeholder_prefix)
    embeddings = textual_trainer.train_embeddings(splits, ti_dir, ti_cfg.steps, ti_cfg.learning_rate)

    token_map: Dict[str, str] = {
        cls: f"{ti_cfg.placeholder_prefix}_{cls}>" for cls in {s.cls_name for s in samples}
    }

    lora_trainer = LoRATrainer(lora_cfg)
    lora_path = lora_trainer.train(splits, token_map, lora_dir)

    extractor = GuidanceExtractor(
        guidance_cfg.hed_repo_id,
        guidance_cfg.midas_repo_id,
        guidance_cfg.hed_ckpt,
        guidance_cfg.midas_ckpt,
    )
    conditioning = extractor.batch_process(samples, guidance_dir)

    generator = ControlNetGenerator(gen_cfg)
    prompt_items = [(s.image_path.stem, gen_cfg.prompt_template.format(token=token_map[s.cls_name])) for s in samples]
    generator.generate(lora_path, prompt_items, conditioning, gen_dir)
