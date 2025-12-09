from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Dict, List, Sequence, Optional

import torch
from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    StableDiffusionControlNetImg2ImgPipeline,
)
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    PNDMScheduler,
)
from PIL import Image
from rich.console import Console
from rich.progress import track
from safetensors.torch import load_file

from .config import GenerationConfig
from .lora_train import AttnProcsLayers, get_lora_processor_class, _estimate_hidden_size  # ensures Module subclass even on older diffusers

SCHEDULER_REGISTRY = {
    "DPMSolverMultistepScheduler": DPMSolverMultistepScheduler,
    "EulerDiscreteScheduler": EulerDiscreteScheduler,
    "EulerAncestralDiscreteScheduler": EulerAncestralDiscreteScheduler,
    "DDIMScheduler": DDIMScheduler,
    "PNDMScheduler": PNDMScheduler,
}

logger = logging.getLogger(__name__)


class ControlNetGenerator:
    """
    Component 4 of the workflow (ControlNet):
    
    ControlNet-based controllable image generation that utilizes:
    1. Original input features (canny edge from original image)
    2. Contour features (HED holistically-nested edge detection)
    3. Depth map features (MiDaS depth estimation)
    
    The ControlNet assists the generative network in achieving controllable 
    generation. As a result, the generated defect positions in the images 
    align closely with the original images, allowing for the reuse of 
    existing annotations and alleviating the substantial annotation effort 
    involved in generating images.
    
    Works in conjunction with:
    - CLIP textual inversion (Component 1): provides prompt keywords
    - LoRA fine-tuned SD 1.5 (Component 2): provides style and texture
    - HED/MiDaS features (Component 3): provides controllable guidance
    """
    def __init__(self, cfg: GenerationConfig):
        self.cfg = cfg

    @staticmethod
    def _safe_empty_cache() -> None:
        """Clear CUDA cache and surface CUDA context failures early."""
        if not torch.cuda.is_available():
            return
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        except Exception as exc:  # noqa: BLE001
            # If CUDA context is corrupted (e.g., illegal memory access), fail fast
            logger.error("CUDA empty_cache failed; aborting run to avoid repeated illegal access: %s", exc)
            raise

    def _require_discrete_cuda(self) -> torch.device:
        """Ensure a CUDA-capable discrete GPU is present and return its device."""
        if not torch.cuda.is_available():
            raise RuntimeError(
                "检测到未启用 CUDA。当前流程已强制使用 NVIDIA 独立显卡，不再回落 CPU/核显。请安装正确的驱动与 CUDA 工具链。"
            )

        # Respect CUDA_VISIBLE_DEVICES if set; default to the current device
        device_idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device_idx)
        logger.info(
            "Using CUDA device %d: %s (%.1f GB)",
            device_idx,
            props.name,
            props.total_memory / (1024 ** 3),
        )
        return torch.device(f"cuda:{device_idx}")

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

    def _build_base_pipe(
        self,
        control_nets: Sequence[ControlNetModel],
        *,
        use_img2img: bool = False,
    ) -> StableDiffusionControlNetPipeline:
        try:
            pipe_cls = (
                StableDiffusionControlNetImg2ImgPipeline if use_img2img else StableDiffusionControlNetPipeline
            )
            pipe = pipe_cls.from_pretrained(
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

    def build_pipeline(
        self,
        lora_path: Path,
        *,
        use_img2img: bool = False,
    ) -> StableDiffusionControlNetPipeline:
        # Enforce discrete GPU usage (no CPU/iGPU fallback)
        device = self._require_discrete_cuda()
        torch.cuda.empty_cache()
        
        logger.info("Building pipeline (img2img=%s) with base model %s", use_img2img, self.cfg.base_model)
        control_nets = self._load_controlnets()
        logger.info("Loaded %d ControlNet(s)", len(control_nets))
        pipe = self._build_base_pipe(control_nets, use_img2img=use_img2img)
        self._configure_scheduler(pipe)
        self._load_lora(pipe, lora_path)
        logger.info("LoRA weights loaded from %s", lora_path)
        
        # Disable xformers to avoid CUDA kernel crashes; rely on PyTorch 2.0+ SDPA which is stable
        try:
            pipe.disable_xformers_memory_efficient_attention()
            logger.info("xformers disabled for stability (using PyTorch SDPA)")
        except Exception:
            pass
        
        # Use model CPU offload for lower VRAM usage and stability
        pipe.enable_model_cpu_offload()
        logger.info("Model CPU offload enabled")
        
        # Set to eval mode for inference
        pipe.unet.eval()
        for cn in pipe.controlnet if isinstance(pipe.controlnet, list) else [pipe.controlnet]:
            cn.eval()
        
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
        """
        Load three types of conditioning images for multi-ControlNet:
        1. Canny edge (original input structure)
        2. HED contour (approximate contours)
        3. MiDaS depth (depth map)
        
        All images are resized to 512x512 to ensure compatibility.
        """
        if stem not in conditioning:
            raise KeyError(f"缺少样本 {stem} 的引导文件。")
        images: List[Image.Image] = []
        entry = conditioning[stem]
        target_size = (512, 512)  # Standard SD resolution
        
        for modality in self.cfg.controlnet_modalities:
            if modality not in entry:
                raise KeyError(f"样本 {stem} 缺少模态 {modality} 的引导图。")
            img = Image.open(entry[modality]).convert("RGB")
            # Resize to ensure all conditioning images have the same dimensions
            if img.size != target_size:
                img = img.resize(target_size, Image.Resampling.LANCZOS)
            images.append(img)
        return images

    def _load_init_image(self, stem: str, init_images: Dict[str, Path]) -> Image.Image:
        if stem not in init_images:
            raise KeyError(f"缺少样本 {stem} 的原始图像，无法执行细节保持 img2img。")
        image_path = Path(init_images[stem])
        if not image_path.exists():
            raise FileNotFoundError(f"找不到用于 img2img 的原始图像: {image_path}")
        img = Image.open(image_path).convert("RGB")
        target_size = (512, 512)
        if img.size != target_size:
            img = img.resize(target_size, Image.Resampling.LANCZOS)
        return img

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate(
        self,
        lora_path: Path,
        prompt_items: List[tuple[str, str]],
        conditioning: Dict[str, Dict[str, Path]],
        output_dir: Path,
        init_images: Optional[Dict[str, Path]] = None,
    ) -> List[Path]:
        self._ensure_lengths()
        use_img2img = bool(init_images)
        pipe = self.build_pipeline(lora_path, use_img2img=use_img2img)
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "Starting generation for %d prompt(s); img2img=%s; output_dir=%s",
            len(prompt_items),
            use_img2img,
            output_dir,
        )

        results: List[Path] = []
        for idx, (stem, prompt) in enumerate(track(prompt_items, description="Generating")):
            logger.info("[%d/%d] Generating '%s'", idx + 1, len(prompt_items), stem)
            try:
                # Clear CUDA cache before each generation; fail fast on CUDA context errors
                self._safe_empty_cache()
                
                ctrl_images = self._load_condition_images(stem, conditioning)
                # If only one ControlNet is in use, we must pass a single image instead of a list
                cond_image = ctrl_images[0] if len(ctrl_images) == 1 else ctrl_images
                
                # Generate with error handling and type safety
                with torch.inference_mode():
                    if use_img2img:
                        if init_images is None:
                            raise RuntimeError("init_images 缺失，无法执行 img2img 以保留纹理细节。")
                        init_image = self._load_init_image(stem, init_images)
                        output = pipe(
                            prompt,
                            image=init_image,
                            control_image=cond_image,
                            num_inference_steps=self.cfg.num_inference_steps,
                            guidance_scale=self.cfg.guidance_scale,
                            strength=self.cfg.denoising_strength,
                            controlnet_conditioning_scale=self.cfg.control_scales,
                        )
                    else:
                        output = pipe(
                            prompt,
                            num_inference_steps=self.cfg.num_inference_steps,
                            guidance_scale=self.cfg.guidance_scale,
                            image=cond_image,
                            controlnet_conditioning_scale=self.cfg.control_scales,
                        )
                    image = output.images[0]
                
                out_path = output_dir / f"{stem}.png"
                image.save(out_path)
                results.append(out_path)
                logger.info("Saved %s", out_path)
                
            except RuntimeError as e:
                # If CUDA context is corrupted, abort the run instead of spamming retries
                if "CUDA error" in str(e) or "illegal memory access" in str(e):
                    console_err = Console(stderr=True)
                    console_err.print(
                        "[red]检测到 CUDA 非法访问/上下文损坏，终止本次生成。请重启 Python 进程并考虑设置 CUDA_LAUNCH_BLOCKING=1 以定位问题。[/red]"
                    )
                    logger.exception("Fatal CUDA error while generating %s", stem)
                    raise

                console_err = Console(stderr=True)
                console_err.print(f"[red]生成 {stem} 时出错: {e}[/red]")
                console_err.print(f"[yellow]跳过该样本并继续...[/yellow]")
                logger.exception("Runtime error while generating %s", stem)
                continue
            except Exception as e:
                console_err = Console(stderr=True)
                console_err.print(f"[red]处理 {stem} 时发生未预期错误: {e}[/red]")
                logger.exception("Unexpected error while generating %s", stem)
                raise

        # Final cleanup: move back to CPU when supported
        try:
            pipe.to("cpu")
        except NotImplementedError as exc:
            logger.warning("Skipping final CPU transfer due to meta tensor limitation: %s", exc)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        logger.info("Generation complete. Produced %d image(s)", len(results))
        return results
