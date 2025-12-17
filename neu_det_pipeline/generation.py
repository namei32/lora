from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Dict, List, Sequence, Optional, Any

import torch
from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    StableDiffusionControlNetImg2ImgPipeline,
    StableDiffusionImg2ImgPipeline,
)
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    PNDMScheduler,
)
from PIL import Image, ImageDraw, ImageFilter
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

try:  # Optional: only required when using bbox inpaint modes
    from diffusers import StableDiffusionInpaintPipeline, StableDiffusionControlNetInpaintPipeline  # type: ignore
except Exception:  # noqa: BLE001
    StableDiffusionInpaintPipeline = None  # type: ignore[assignment]
    StableDiffusionControlNetInpaintPipeline = None  # type: ignore[assignment]


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
        use_inpaint: bool = False,
    ) -> Any:
        # Inpaint pipelines (mask inside repaint; outside locked)
        if use_inpaint:
            if len(control_nets) == 0:
                if StableDiffusionInpaintPipeline is None:
                    raise RuntimeError(
                        "当前 diffusers 环境不支持 StableDiffusionInpaintPipeline。"
                        "请升级 diffusers，或改用 img2img+后处理混合。"
                    )
                try:
                    pipe = StableDiffusionInpaintPipeline.from_pretrained(  # type: ignore[misc]
                        self.cfg.base_model,
                        torch_dtype=torch.float16,
                        safety_checker=None,
                        use_auth_token=self._auth_token(),
                    )
                except OSError as exc:
                    hint = "建议使用 inpainting 版本基础模型（例如 runwayml/stable-diffusion-inpainting）。"
                    raise OSError(f"无法加载 inpaint 基础模型 '{self.cfg.base_model}': {exc}. {hint}") from exc
                return pipe

            if StableDiffusionControlNetInpaintPipeline is None:
                raise RuntimeError(
                    "当前 diffusers 环境不支持 StableDiffusionControlNetInpaintPipeline。"
                    "请升级 diffusers。"
                )
            try:
                pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(  # type: ignore[misc]
                    self.cfg.base_model,
                    controlnet=list(control_nets),
                    torch_dtype=torch.float16,
                    safety_checker=None,
                    use_auth_token=self._auth_token(),
                )
            except OSError as exc:
                hint = "建议使用 inpainting 版本基础模型（例如 runwayml/stable-diffusion-inpainting）。"
                raise OSError(f"无法加载 ControlNet inpaint 基础模型 '{self.cfg.base_model}': {exc}. {hint}") from exc
            return pipe

        # If no ControlNet is requested (A1), fall back to plain SD img2img pipeline
        if len(control_nets) == 0:
            try:
                pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                    self.cfg.base_model,
                    torch_dtype=torch.float16,
                    safety_checker=None,
                    use_auth_token=self._auth_token(),
                )
            except OSError as exc:
                hint = "请确保 base_model 指向合法的 Stable Diffusion 目录 (含 model_index.json / unet / vae 等)。"
                raise OSError(f"无法加载基础模型 '{self.cfg.base_model}': {exc}. {hint}") from exc
            return pipe  # type: ignore[return-value]

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

    def _load_textual_inversion_embeddings(self, pipe: Any, embeddings_dir: Path) -> None:
        """
        Load textual inversion embeddings saved by our trainer (dict with 'token'/'embedding').
        This enables prompts containing <neu_xxx> tokens to actually influence generation.
        """
        try:
            tokenizer = pipe.tokenizer
            text_encoder = pipe.text_encoder
        except Exception as exc:  # noqa: BLE001
            logger.warning("Pipeline lacks tokenizer/text_encoder; skipping textual inversion load: %s", exc)
            return

        embeddings_dir = Path(embeddings_dir)
        embedding_files = list(embeddings_dir.glob("*_embedding.pt"))
        if not embedding_files:
            logger.info("No textual inversion embeddings found in %s", embeddings_dir)
            return

        logger.info("Loading %d textual inversion embedding(s) from %s", len(embedding_files), embeddings_dir)
        loaded = 0
        for emb_file in embedding_files:
            cls_name = emb_file.stem.replace("_embedding", "")
            token = f"<neu_{cls_name}>"
            checkpoint = torch.load(emb_file, map_location=text_encoder.device)
            if isinstance(checkpoint, dict) and "embedding" in checkpoint:
                embedding = checkpoint["embedding"]
            else:
                embedding = checkpoint
            if not hasattr(embedding, "shape"):
                logger.warning("Invalid embedding file %s; skipping", emb_file)
                continue
            embedding = embedding.to(text_encoder.device)

            num_added = tokenizer.add_tokens(token)
            if num_added == 0:
                # Token exists; still ensure embedding is set (overwrites to be safe)
                token_id = tokenizer.convert_tokens_to_ids(token)
                try:
                    text_encoder.get_input_embeddings().weight.data[token_id] = embedding
                    loaded += 1
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Failed to set existing token embedding %s: %s", token, exc)
                continue

            text_encoder.resize_token_embeddings(len(tokenizer))
            token_id = tokenizer.convert_tokens_to_ids(token)
            text_encoder.get_input_embeddings().weight.data[token_id] = embedding
            loaded += 1

        logger.info("Loaded %d textual inversion token(s)", loaded)

    def build_pipeline(
        self,
        lora_path: Path,
        *,
        use_img2img: bool = False,
        use_inpaint: bool = False,
    ) -> Any:
        # Enforce discrete GPU usage (no CPU/iGPU fallback)
        device = self._require_discrete_cuda()
        torch.cuda.empty_cache()
        
        logger.info("Building pipeline (img2img=%s) with base model %s", use_img2img, self.cfg.base_model)
        control_nets = self._load_controlnets()
        logger.info("Loaded %d ControlNet(s)", len(control_nets))
        pipe = self._build_base_pipe(control_nets, use_img2img=use_img2img, use_inpaint=use_inpaint)
        self._configure_scheduler(pipe)
        # Auto-load textual inversion embeddings if present next to LoRA output
        # Typical layout: outputs/lora/lora.safetensors and outputs/textual_inversion/*_embedding.pt
        try:
            embeddings_dir = Path(lora_path).parent.parent / "textual_inversion"
            if embeddings_dir.exists():
                self._load_textual_inversion_embeddings(pipe, embeddings_dir)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to auto-load textual inversion embeddings: %s", exc)
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
        if hasattr(pipe, "controlnet"):
            controlnets = pipe.controlnet if isinstance(pipe.controlnet, list) else [pipe.controlnet]
            for cn in controlnets:
                if cn is None:
                    continue
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

    def _load_init_image(self, stem: str, init_images: Dict[str, Path]) -> tuple[Image.Image, tuple[int, int]]:
        if stem not in init_images:
            raise KeyError(f"缺少样本 {stem} 的原始图像，无法执行细节保持 img2img。")
        image_path = Path(init_images[stem])
        if not image_path.exists():
            raise FileNotFoundError(f"找不到用于 img2img 的原始图像: {image_path}")
        img = Image.open(image_path).convert("RGB")
        orig_size = img.size
        target_size = (512, 512)
        if img.size != target_size:
            img = img.resize(target_size, Image.Resampling.LANCZOS)
        return img, orig_size

    # ------------------------------------------------------------------
    # Mask utilities
    # ------------------------------------------------------------------
    def _make_soft_mask(
        self,
        bbox: tuple[int, int, int, int],
        target_size: tuple[int, int],
        original_size: tuple[int, int],
    ) -> Image.Image:
        """
        Build a soft mask (white=replace, black=keep original) from bbox with margin and feather.
        """
        width, height = target_size
        orig_w, orig_h = original_size
        xmin, ymin, xmax, ymax = bbox

        # Scale bbox from original resolution to target resolution
        scale_x = width / max(orig_w, 1)
        scale_y = height / max(orig_h, 1)
        xmin = int(xmin * scale_x)
        xmax = int(xmax * scale_x)
        ymin = int(ymin * scale_y)
        ymax = int(ymax * scale_y)

        margin_x = int((xmax - xmin) * self.cfg.mask_margin_ratio)
        margin_y = int((ymax - ymin) * self.cfg.mask_margin_ratio)
        x0 = max(0, xmin - margin_x)
        y0 = max(0, ymin - margin_y)
        x1 = min(width, xmax + margin_x)
        y1 = min(height, ymax + margin_y)

        mask = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask)
        draw.rectangle([x0, y0, x1, y1], fill=255)

        radius = max(width, height) * self.cfg.mask_feather_ratio
        if radius > 0:
            mask = mask.filter(ImageFilter.GaussianBlur(radius=radius))
        return mask

    def _blend_with_mask(
        self,
        base_img: Image.Image,
        gen_img: Image.Image,
        mask: Image.Image,
    ) -> Image.Image:
        """
        Composite generated image onto base image using mask (white=gen, black=base).
        """
        base_rgb = base_img.convert("RGB")
        gen_rgb = gen_img.convert("RGB")
        mask_l = mask.convert("L")
        return Image.composite(gen_rgb, base_rgb, mask_l)

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
        bbox_map: Optional[Dict[str, tuple[int, int, int, int]]] = None,
        use_bbox_mask: bool = True,
    ) -> List[Path]:
        self._ensure_lengths()
        use_img2img = bool(init_images)
        use_inpaint = bool(use_bbox_mask and bbox_map and init_images)
        if use_inpaint and init_images is None:
            raise RuntimeError("启用 bbox mask 时必须提供 init_images（inpaint 需要原图）。")
        pipe = self.build_pipeline(lora_path, use_img2img=use_img2img, use_inpaint=use_inpaint)
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "Starting generation for %d prompt(s); img2img=%s; inpaint=%s; output_dir=%s",
            len(prompt_items),
            use_img2img,
            use_inpaint,
            output_dir,
        )

        has_control = len(self.cfg.controlnet_models) > 0
        results: List[Path] = []
        for idx, (stem, prompt) in enumerate(track(prompt_items, description="Generating")):
            logger.info("[%d/%d] Generating '%s'", idx + 1, len(prompt_items), stem)
            try:
                # Clear CUDA cache before each generation; fail fast on CUDA context errors
                self._safe_empty_cache()
                
                cond_image = None
                if has_control:
                    ctrl_images = self._load_condition_images(stem, conditioning)
                    # Always pass as list: required for multiple ControlNets, and safe for single ControlNet
                    # _load_condition_images always returns a list, but ensure it's definitely a list
                    cond_image = list(ctrl_images) if ctrl_images else []
                
                # Generate with error handling and type safety
                with torch.inference_mode():
                    if use_img2img:
                        if init_images is None:
                            raise RuntimeError("init_images 缺失，无法执行 img2img 以保留纹理细节。")
                        init_image, orig_size = self._load_init_image(stem, init_images)

                        if use_inpaint:
                            if bbox_map is None or stem not in bbox_map:
                                raise KeyError(f"缺少样本 {stem} 的 bbox，无法执行 inpaint。")
                            mask = self._make_soft_mask(bbox_map[stem], init_image.size, orig_size)
                            if has_control:
                                output = pipe(
                                    prompt,
                                    image=init_image,
                                    mask_image=mask,
                                    control_image=cond_image,
                                    num_inference_steps=self.cfg.num_inference_steps,
                                    guidance_scale=self.cfg.guidance_scale,
                                    strength=self.cfg.denoising_strength,
                                    controlnet_conditioning_scale=self.cfg.control_scales,
                                )
                            else:
                                output = pipe(
                                    prompt,
                                    image=init_image,
                                    mask_image=mask,
                                    num_inference_steps=self.cfg.num_inference_steps,
                                    guidance_scale=self.cfg.guidance_scale,
                                    strength=self.cfg.denoising_strength,
                                )
                        else:
                            if has_control:
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
                                    image=init_image,
                                    num_inference_steps=self.cfg.num_inference_steps,
                                    guidance_scale=self.cfg.guidance_scale,
                                    strength=self.cfg.denoising_strength,
                                )
                    else:
                        if has_control:
                            output = pipe(
                                prompt,
                                num_inference_steps=self.cfg.num_inference_steps,
                                guidance_scale=self.cfg.guidance_scale,
                                image=cond_image,
                                controlnet_conditioning_scale=self.cfg.control_scales,
                            )
                        else:
                            output = pipe(
                                prompt,
                                num_inference_steps=self.cfg.num_inference_steps,
                                guidance_scale=self.cfg.guidance_scale,
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
