from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass(slots=True)
class DatasetConfig:
    root: Path
    test_size: float = 0.1
    seed: int = 42

    def resolve(self) -> Path:
        return self.root.expanduser().resolve()


@dataclass(slots=True)
class TextualInversionConfig:
    model_id: str = "runwayml/stable-diffusion-v1-5"
    placeholder_prefix: str = "<neu"
    initializer_token: str = "steel"
    steps: int = 800
    batch_size: int = 4
    resolution: int = 512
    learning_rate: float = 5e-4
    mixed_precision: str = "fp16"
    max_grad_norm: float = 1.0
    lr_scheduler: str = "cosine"
    lr_warmup_steps: int = 100
    gradient_accumulation_steps: int = 1
    prompt_template: str = "macro photo of {token} steel defect"


@dataclass(slots=True)
class LoRAConfig:
    model_id: str = "runwayml/stable-diffusion-v1-5"
    rank: int = 8
    alpha: int = 16
    learning_rate: float = 1e-4
    steps: int = 1000
    batch_size: int = 2
    resolution: int = 512
    mixed_precision: str = "fp16"
    max_grad_norm: float = 1.0
    lr_scheduler: str = "cosine"
    lr_warmup_steps: int = 100
    gradient_accumulation_steps: int = 1
    prompt_template: str = "macro shot of {token} steel surface"


@dataclass(slots=True)
class GuidanceConfig:
    hed_repo_id: str = "lllyasviel/Annotators"
    midas_repo_id: str = "lllyasviel/Annotators"
    hed_ckpt: str = "ControlNetHED.pth"
    midas_ckpt: str = "dpt_hybrid-midas-501f0c75.pt"


@dataclass(slots=True)
class GenerationConfig:
    base_model: str = "runwayml/stable-diffusion-v1-5"
    prompt_template: str = "macro shot of {token} steel surface"
    hf_token: str | None = None
    controlnet_models: List[str] = field(
        default_factory=lambda: [
            "lllyasviel/control_v11p_sd15_canny",
            "lllyasviel/control_v11p_sd15_softedge",
            "lllyasviel/control_v11f1p_sd15_depth",
        ]
    )
    controlnet_modalities: List[str] = field(default_factory=lambda: ["hed", "hed", "depth"])
    num_inference_steps: int = 30
    guidance_scale: float = 7.5
    control_scales: List[float] = field(default_factory=lambda: [0.8, 0.8, 0.9])
    scheduler: str = "DPMSolverMultistepScheduler"
