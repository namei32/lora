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
    """
    Complete controllable generative pipeline based on Stable Diffusion.
    
    This pipeline consists of four main components following the workflow:
    
    1. **CLIP + Textual Inversion**: 
       Employs the textual inversion feature of the Contrastive Language-Image 
       Pre-training (CLIP) model to generate prompt keywords corresponding to 
       each category of the original images.
    
    2. **LoRA Fine-tuning**: 
       The Stable Diffusion 1.5 generation model is fine-tuned using Low-rank 
       adaptation (LoRA) on the defect dataset, enabling it to learn the style, 
       texture, and other characteristics of steel defects.
    
    3. **HED & MiDaS Guidance Extraction**: 
       By leveraging the HED (Holistically-nested Edge Detection) and MiDaS 
       (depth map estimation) models, approximate contours and depth maps of 
       the images are obtained. Additionally, Canny edge detection extracts 
       the original input structure.
    
    4. **ControlNet Controllable Generation**: 
       The ControlNet, utilizing the original input features (canny edges), 
       contour features (HED), and depth map features (MiDaS), assists the 
       generative network in achieving controllable generation. As a result, 
       the generated defect positions in the images align closely with the 
       original images, allowing for the reuse of existing annotations and 
       alleviating the substantial annotation effort involved in generating images.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    samples = collect_dataset(dataset_cfg.resolve())
    splits = split_dataset(samples, dataset_cfg.test_size, dataset_cfg.seed)

    ti_dir = work_dir / "textual_inversion"
    lora_dir = work_dir / "lora"
    guidance_dir = work_dir / "guidance"
    gen_dir = work_dir / "generated"

    # Step 1: Textual Inversion with CLIP to generate prompt keywords per category
    textual_trainer = TextualInversionTrainer(model_id=ti_cfg.model_id, token_prefix=ti_cfg.placeholder_prefix)
    embeddings = textual_trainer.train_embeddings(splits, ti_dir, ti_cfg.steps, ti_cfg.learning_rate)

    token_map: Dict[str, str] = {
        cls: f"{ti_cfg.placeholder_prefix}_{cls}>" for cls in {s.cls_name for s in samples}
    }

    # Step 2: LoRA fine-tuning to learn steel defect style and texture characteristics
    lora_trainer = LoRATrainer(lora_cfg)
    lora_path = lora_trainer.train(splits, token_map, lora_dir)

    # Step 3: Extract three types of guidance features:
    #   - Canny edges from original images (original input structure)
    #   - HED contours (approximate contours)
    #   - MiDaS depth maps (depth structure)
    extractor = GuidanceExtractor(
        guidance_cfg.hed_repo_id,
        guidance_cfg.midas_repo_id,
        guidance_cfg.hed_ckpt,
        guidance_cfg.midas_ckpt,
    )
    conditioning = extractor.batch_process(samples, guidance_dir)

    # Step 4: ControlNet controllable generation using all three feature types
    #   to ensure generated defect positions align with original images,
    #   allowing reuse of existing annotations
    generator = ControlNetGenerator(gen_cfg)
    prompt_items = [(s.image_path.stem, gen_cfg.prompt_template.format(token=token_map[s.cls_name])) for s in samples]
   init_images = {s.image_path.stem: s.image_path for s in samples}
   generator.generate(lora_path, prompt_items, conditioning, gen_dir, init_images=init_images)
