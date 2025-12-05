from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


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
    # Three ControlNets for: 1) original input features (canny edge from original),
    # 2) contour features (HED-based soft edge), 3) depth map features (MiDaS)
    controlnet_models: List[str] = field(
        default_factory=lambda: [
            "lllyasviel/control_v11p_sd15_canny",      # Original input structure
            "lllyasviel/control_v11p_sd15_softedge",   # HED contour features
            "lllyasviel/control_v11f1p_sd15_depth",    # MiDaS depth features
        ]
    )
    # Modalities correspond to guidance features: canny from original, HED contours, depth maps
    controlnet_modalities: List[str] = field(default_factory=lambda: ["canny", "hed", "depth"])
    num_inference_steps: int = 50  # SD default
    guidance_scale: float = 7.5  # SD default
    # Control scales: Canny default, HED 1.0, MiDaS 1.0 as per requirements
    control_scales: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    control_start_steps: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    control_end_steps: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    denoising_strength: float = 0.2  # As per requirements
    scheduler: str = "PNDMScheduler"  # SD default scheduler
    seed: int = 42


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


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"配置文件 {path} 的顶层必须是字典。")
    return data


def load_config_bundle(config_path: Optional[Path] = None) -> ConfigBundle:
    resolved_path: Optional[Path]
    if config_path is not None:
        resolved_path = config_path.expanduser().resolve()
        if not resolved_path.exists():
            raise FileNotFoundError(f"未找到配置文件: {resolved_path}")
    else:
        resolved_path = DEFAULT_CONFIG_PATH if DEFAULT_CONFIG_PATH.exists() else None

    defaults: Dict[str, Any] = {}
    if resolved_path is not None:
        defaults = _load_yaml(resolved_path).get("defaults", {})

    dataset_cfg = _instantiate_config(DatasetConfig, defaults.get("dataset"))
    textual_cfg = _instantiate_config(TextualInversionConfig, defaults.get("textual_inversion"))
    lora_cfg = _instantiate_config(LoRAConfig, defaults.get("lora"))
    guidance_cfg = _instantiate_config(GuidanceConfig, defaults.get("guidance"))
    generation_cfg = _instantiate_config(GenerationConfig, defaults.get("generation"))

    return ConfigBundle(
        dataset=dataset_cfg,
        textual_inversion=textual_cfg,
        lora=lora_cfg,
        guidance=guidance_cfg,
        generation=generation_cfg,
        source_path=resolved_path,
    )
