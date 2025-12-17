from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml


@dataclass
class DatasetConfig:
    root: Path = field(default_factory=lambda: Path("NEU-DET"))
    test_size: float = 0.1
    seed: int = 42

    def resolve(self) -> Path:
        return self.root.expanduser().resolve()


@dataclass
class TextualInversionConfig:
    model_id: str = "runwayml/stable-diffusion-v1-5"
    placeholder_prefix: str = "<neu"
    initializer_token: str = "steel"
    steps: int = 800
    batch_size: int = 4
    resolution: int = 512
    learning_rate: float = 5e-4
    mixed_precision: str = "no"  # Changed from fp16 to avoid Float/Half type errors
    max_grad_norm: float = 1.0
    lr_scheduler: str = "cosine"
    lr_warmup_steps: int = 100
    gradient_accumulation_steps: int = 1
    prompt_template: str = "macro photo of {token} steel defect"


@dataclass
class LoRAConfig:
    model_id: str = "runwayml/stable-diffusion-v1-5"
    rank: int = 8
    alpha: int = 8  # alpha = rank for LoRA weight 1.0
    dropout_rate: float = 0.0
    learning_rate: float = 1e-5  # 降低学习率防止发散
    steps: int = 100  # 增加训练步数
    batch_size: int = 1  # 降低到最小值避免 OOM
    resolution: int = 512
    mixed_precision: str = "no"  # Changed from fp16 to avoid Float/Half type errors
    max_grad_norm: float = 1.0
    lr_scheduler: str = "cosine"
    lr_warmup_steps: int = 20  # 占总步数20%
    gradient_accumulation_steps: int = 4  # 增加梯度累积补偿小 batch（有效 batch=4）
    prompt_template: str = "macro shot of {token} steel surface"
    seed: int = 42


@dataclass
class GuidanceConfig:
    hed_repo_id: str = "lllyasviel/Annotators"
    midas_repo_id: str = "lllyasviel/Annotators"
    hed_ckpt: str = "ControlNetHED.pth"
    midas_ckpt: str = "dpt_hybrid-midas-501f0c75.pt"


@dataclass
class GenerationConfig:
    base_model: str = "runwayml/stable-diffusion-v1-5"
    prompt_template: str = "a photo of {token}"
    hf_token: str | None = None
    # Soft mask parameters for bbox-constrained blending
    mask_margin_ratio: float = 0.15  # expand bbox by 15% on each side
    mask_feather_ratio: float = 0.08  # feather radius relative to image size
    # Two ControlNets as per paper: 1) HED contour features, 2) depth map features (MiDaS)
    controlnet_models: List[str] = field(
        default_factory=lambda: [
            "lllyasviel/control_v11p_sd15_softedge",   # HED contour features
            "lllyasviel/control_v11f1p_sd15_depth",    # MiDaS depth features
        ]
    )
    # Modalities correspond to guidance features: HED contours, depth maps
    controlnet_modalities: List[str] = field(default_factory=lambda: ["hed", "depth"])
    num_inference_steps: int = 40  # Inference steps
    guidance_scale: float = 7.0  # Per paper: CFG scale 7.0
    # Control scales: HED 0.7, MiDaS 0.7 (ControlNet weights)
    control_scales: List[float] = field(default_factory=lambda: [1.0, 0.5])
    control_start_steps: List[float] = field(default_factory=lambda: [0.0, 0.0])
    control_end_steps: List[float] = field(default_factory=lambda: [1.0, 1.0])
    denoising_strength: float = 0.2 # Denoising strength
    scheduler: str = "DPMSolverMultistepScheduler"  # DPM++ 2 M Karras (per paper)
    seed: int = 865985793  # Paper-specified seed for reproducibility


@dataclass
class ConfigBundle:
    dataset: DatasetConfig
    textual_inversion: TextualInversionConfig
    lora: LoRAConfig
    guidance: GuidanceConfig
    generation: GenerationConfig
    source_path: Optional[Path] = None


def _is_path_type(tp: Any) -> bool:
    if tp is Path:
        return True
    origin = getattr(tp, "__origin__", None)
    if origin is Union:
        return any(arg is Path for arg in getattr(tp, "__args__", ()))
    return False


def _instantiate_config(cls: type, overrides: Optional[Dict[str, Any]]) -> Any:
    if not overrides:
        return cls()
    init_kwargs: Dict[str, Any] = {}
    for field_info in fields(cls):
        if field_info.name not in overrides:
            continue
        value = overrides[field_info.name]
        if _is_path_type(field_info.type) and isinstance(value, str):
            value = Path(value)
        init_kwargs[field_info.name] = value
    return cls(**init_kwargs)


def load_config_bundle(config_path: Optional[Path] = None) -> ConfigBundle:
    """
    Load configuration from dataclass defaults only.
    YAML config files are no longer supported.
    All configuration is defined in config.py dataclasses.
    
    Args:
        config_path: Deprecated, ignored for backwards compatibility
    """
    if config_path is not None:
        import warnings
        warnings.warn(
            "config_path parameter is deprecated. "
            "Configuration is now loaded from config.py dataclass defaults only.",
            DeprecationWarning,
            stacklevel=2
        )

    # Use dataclass defaults directly
    dataset_cfg = DatasetConfig()
    textual_cfg = TextualInversionConfig()
    lora_cfg = LoRAConfig()
    guidance_cfg = GuidanceConfig()
    generation_cfg = GenerationConfig()

    return ConfigBundle(
        dataset=dataset_cfg,
        textual_inversion=textual_cfg,
        lora=lora_cfg,
        guidance=guidance_cfg,
        generation=generation_cfg,
        source_path=None,  # No YAML file source
    )
