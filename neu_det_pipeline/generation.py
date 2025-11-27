from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Sequence, Optional

import torch
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
)
from PIL import Image
from rich.progress import track
from safetensors.torch import load_file

from .config import GenerationConfig
from .lora_train import AttnProcsLayers, get_lora_processor_class, _estimate_hidden_size  # ensures Module subclass even on older diffusers

SCHEDULER_REGISTRY = {
    "DPMSolverMultistepScheduler": DPMSolverMultistepScheduler,
    "EulerDiscreteScheduler": EulerDiscreteScheduler,
    "EulerAncestralDiscreteScheduler": EulerAncestralDiscreteScheduler,
    "DDIMScheduler": DDIMScheduler,
}


class ControlNetGenerator:
    def __init__(self, cfg: GenerationConfig):
        self.cfg = cfg

    # ------------------------------------------------------------------
    # Loading helpers
    # ------------------------------------------------------------------
    def _auth_token(self) -> Optional[str]:
        return self.cfg.hf_token or os.getenv("HF_TOKEN")

    def _load_controlnets(self) -> List[ControlNetModel]:
        models: List[ControlNetModel] = []
        for repo_id in track(self.cfg.controlnet_models, description="Loading ControlNets"):
            try:
                models.append(
                    ControlNetModel.from_pretrained(
                        repo_id,
                        torch_dtype=torch.float16,
                        use_auth_token=self._auth_token(),
                    )
                )
            except OSError as exc:
                hint = (
                    "请确认 ControlNet 模型已下载或设置 HF_TOKEN，以便 diffusers 可以访问对应权重。"
                )
                raise OSError(f"无法加载 ControlNet '{repo_id}': {exc}. {hint}") from exc
        return models

    def _build_base_pipe(self, control_nets: Sequence[ControlNetModel]) -> StableDiffusionControlNetPipeline:
        try:
            pipe = StableDiffusionControlNetPipeline.from_pretrained(
                self.cfg.base_model,
                controlnet=list(control_nets),
                torch_dtype=torch.float16,
                safety_checker=None,
                use_auth_token=self._auth_token(),
            )
        except OSError as exc:
            hint = "请确保 base_model 指向合法的 Stable Diffusion 目录 (含 model_index.json / unet / vae 等)。"
            raise OSError(f"无法加载基础模型 '{self.cfg.base_model}': {exc}. {hint}") from exc
        return pipe

    def _configure_scheduler(self, pipe: StableDiffusionControlNetPipeline) -> None:
        scheduler_cls = SCHEDULER_REGISTRY.get(self.cfg.scheduler)
        if scheduler_cls is None:
            available = ", ".join(SCHEDULER_REGISTRY)
            raise ValueError(f"未知 scheduler: {self.cfg.scheduler}. 可选值: {available}")
        pipe.scheduler = scheduler_cls.from_config(pipe.scheduler.config)

    def _ensure_module_processors(self, pipe: StableDiffusionControlNetPipeline) -> None:
        """Ensures every attention processor registered on the UNet is a Module."""
        updated: Dict[str, torch.nn.Module] = {}
        needs_update = False

        class _AttnModuleWrapper(torch.nn.Module):
            def __init__(self, processor):
                super().__init__()
                self.processor = processor

            def forward(self, *args, **kwargs):  # pragma: no cover - simple proxy
                return self.processor(*args, **kwargs)

        for name, processor in pipe.unet.attn_processors.items():
            if isinstance(processor, torch.nn.Module):
                updated[name] = processor
                continue
            updated[name] = _AttnModuleWrapper(processor)
            needs_update = True

        if needs_update:
            pipe.unet.set_attn_processor(updated)

    def _load_lora(self, pipe: StableDiffusionControlNetPipeline, lora_path: Path) -> None:
        lora_path = Path(lora_path)
        if not lora_path.exists():
            raise FileNotFoundError(f"未找到 LoRA 权重文件: {lora_path}")
        repo_dir = lora_path.parent
        weight_name = lora_path.name
        last_error: Optional[Exception] = None
        # Try native diffusers helpers first (handles adapters exported elsewhere)
        try:
            pipe.load_lora_weights(str(repo_dir), weight_name=weight_name)
            return
        except Exception as exc:  # noqa: BLE001 - capture for fallback
            last_error = exc
        try:
            pipe.unet.load_attn_procs(str(repo_dir), weight_name=weight_name)
            return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
        # Fallback: state dict injection for weights produced by our trainer
        try:
            state = load_file(str(lora_path), device="cpu")

            # Reconstruct the attention processors from the state dict
            attn_procs: Dict[str, torch.nn.Module] = {}
            lora_cls = get_lora_processor_class()

            # Use actual processor names from the UNet, not config
            attn_processor_keys = list(pipe.unet.attn_processors.keys())

            for name in attn_processor_keys:
                cross_attention_dim = pipe.unet.config.cross_attention_dim if name.endswith("attn2.processor") else None
                # Estimate sizes from saved tensors
                proc_state_dict = {}
                safe_name = name.replace(".", "_")
                for key, value in state.items():
                    head, sep, tail = key.partition(".")
                    if not sep:
                        continue
                    if head == safe_name or head.startswith(safe_name + "_"):
                        proc_state_dict[tail] = value

                if not proc_state_dict:
                    continue

                # Infer rank and dims
                inferred_rank: Optional[int] = None
                inferred_hidden: Optional[int] = None
                inferred_cross: Optional[int] = None
                for k, v in proc_state_dict.items():
                    if not hasattr(v, "shape"):
                        continue
                    if k.endswith("down.weight") and len(v.shape) == 2:
                        r, dim = int(v.shape[0]), int(v.shape[1])
                        inferred_rank = inferred_rank or r
                        lower = k.lower()
                        if "_q." in lower or "to_q" in lower:
                            inferred_hidden = inferred_hidden or dim
                        elif "_k." in lower or "to_k" in lower:
                            inferred_cross = inferred_cross or dim
                        elif "_v." in lower or "to_v" in lower:
                            inferred_cross = inferred_cross or dim
                        elif "out." in lower or "to_out" in lower:
                            inferred_hidden = inferred_hidden or dim
                rank = inferred_rank or 4
                hidden_size = inferred_hidden or _estimate_hidden_size(name, pipe.unet.config.block_out_channels)
                cross_dim = inferred_cross if (inferred_cross is not None) else cross_attention_dim

                # Create processor with inferred dimensions
                try:
                    lora_processor = lora_cls(
                        hidden_size=hidden_size,
                        cross_attention_dim=cross_dim,
                        rank=rank,
                        alpha=rank,
                    )
                except TypeError:
                    lora_processor = lora_cls(
                        hidden_size=hidden_size,
                        cross_attention_dim=cross_dim,
                        r=rank,  # type: ignore[arg-type]
                        network_alpha=rank,  # type: ignore[arg-type]
                    )

                lora_processor.load_state_dict(proc_state_dict, strict=False)
                attn_procs[name] = lora_processor

            if not attn_procs:
                raise RuntimeError("Could not find any valid LoRA layers in the state dictionary.")

            pipe.unet.set_attn_processor(attn_procs)

        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"无法加载 LoRA 权重 {lora_path}: {exc}\n最近一次 diffusers 加载错误: {last_error}"
            ) from exc

    def build_pipeline(self, lora_path: Path) -> StableDiffusionControlNetPipeline:
        control_nets = self._load_controlnets()
        pipe = self._build_base_pipe(control_nets)
        self._configure_scheduler(pipe)
        self._load_lora(pipe, lora_path)
        pipe.enable_model_cpu_offload()
        return pipe

    # ------------------------------------------------------------------
    # Conditioning utilities
    # ------------------------------------------------------------------
    def _ensure_lengths(self) -> None:
        n = len(self.cfg.controlnet_models)
        if len(self.cfg.controlnet_modalities) != n:
            raise ValueError("controlnet_modalities 数量需与 controlnet_models 保持一致。")
        if len(self.cfg.control_scales) != n:
            raise ValueError("control_scales 数量需与 controlnet_models 保持一致。")

    def _load_condition_images(
        self,
        stem: str,
        conditioning: Dict[str, Dict[str, Path]],
    ) -> List[Image.Image]:
        if stem not in conditioning:
            raise KeyError(f"缺少样本 {stem} 的引导文件。")
        images: List[Image.Image] = []
        entry = conditioning[stem]
        for modality in self.cfg.controlnet_modalities:
            if modality not in entry:
                raise KeyError(f"样本 {stem} 缺少模态 {modality} 的引导图。")
            images.append(Image.open(entry[modality]).convert("RGB"))
        return images

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate(
        self,
        lora_path: Path,
        prompt_items: List[tuple[str, str]],
        conditioning: Dict[str, Dict[str, Path]],
        output_dir: Path,
    ) -> List[Path]:
        self._ensure_lengths()
        pipe = self.build_pipeline(lora_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        results: List[Path] = []
        for stem, prompt in track(prompt_items, description="Generating"):
            ctrl_images = self._load_condition_images(stem, conditioning)
            # If only one ControlNet is in use, we must pass a single image instead of a list
            cond_image = ctrl_images[0] if len(ctrl_images) == 1 else ctrl_images
            image = pipe(
                prompt,
                num_inference_steps=self.cfg.num_inference_steps,
                guidance_scale=self.cfg.guidance_scale,
                image=cond_image,
                controlnet_conditioning_scale=self.cfg.control_scales,
            ).images[0]
            out_path = output_dir / f"{stem}.png"
            image.save(out_path)
            results.append(out_path)

        pipe.to("cpu")
        torch.cuda.empty_cache()
        return results
