from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from packaging import version
from tqdm import tqdm
from torch.utils.data import DataLoader

from diffusers import StableDiffusionPipeline
from diffusers.schedulers import DDPMScheduler
from diffusers.optimization import get_scheduler

from safetensors.torch import save_file

from .data import DatasetSplits
from .config import LoRAConfig


@dataclass
class LoRASample:
    pixel_values: torch.Tensor
    prompt: str


class LoRADataset:
    def __init__(self, samples: List[object], token_map: Dict[str, str], prompt_template: str, resolution: int):
        self.samples = samples
        self.token_map = token_map
        self.prompt_template = prompt_template
        self.resolution = resolution

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> LoRASample:
        # Lazy import to avoid hard dependency here
        from .data import load_image
        import numpy as np
        import cv2

        sample = self.samples[idx]
        img = load_image(sample.image_path)
        # center-crop and resize to resolution
        h, w = img.shape[:2]
        s = min(h, w)
        y0 = (h - s) // 2
        x0 = (w - s) // 2
        img = img[y0:y0 + s, x0:x0 + s]
        img = cv2.resize(img, (self.resolution, self.resolution), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)
        prompt = self.prompt_template.format(token=self.token_map.get(sample.cls_name, sample.cls_name))
        return LoRASample(pixel_values=img, prompt=prompt)


class LoRATrainer:
    """
    LoRA (Low-Rank Adaptation) trainer for Stable Diffusion fine-tuning.
    
    This implements the second component of the workflow:
    The Stable Diffusion 1.5 generation model is fine-tuned using Low-rank 
    adaptation (LoRA) on the defect dataset, enabling it to learn the style, 
    texture, and other characteristics of steel defects.
    
    LoRA achieves efficient fine-tuning by adding low-rank update matrices 
    to the attention layers, preserving the base model while learning 
    domain-specific features.
    """
    def __init__(self, cfg: LoRAConfig):
        self.cfg = cfg

    def prepare_pipeline(self, token_embeddings_dir: Path = None) -> StableDiffusionPipeline:
        """
        Prepare SD pipeline and load textual inversion embeddings if available.
        
        Args:
            token_embeddings_dir: Directory containing textual inversion embeddings (*.pt files)
        """
        pipe = StableDiffusionPipeline.from_pretrained(
            self.cfg.model_id,
            torch_dtype=torch.float32,
            safety_checker=None,
            requires_safety_checker=False
        )
        pipe.to("cuda" if torch.cuda.is_available() else "cpu")
        pipe.set_progress_bar_config(disable=True)
        
        # Load textual inversion embeddings if directory is provided
        if token_embeddings_dir and token_embeddings_dir.exists():
            self._load_textual_inversion_embeddings(pipe, token_embeddings_dir)
        
        return pipe
    
    def _load_textual_inversion_embeddings(self, pipe: StableDiffusionPipeline, embeddings_dir: Path):
        """Load all textual inversion embeddings from directory."""
        embedding_files = list(embeddings_dir.glob("*_embedding.pt"))
        if not embedding_files:
            print(f"Warning: No embedding files found in {embeddings_dir}")
            return
        
        print(f"Loading {len(embedding_files)} textual inversion embeddings...")
        for emb_file in embedding_files:
            # Extract class name from filename (e.g., "crazing_embedding.pt" -> "crazing")
            cls_name = emb_file.stem.replace("_embedding", "")
            token = f"<neu_{cls_name}>"
            
            # Load embedding (it's saved as a dict with 'token', 'embedding', 'token_id')
            checkpoint = torch.load(emb_file, map_location=pipe.text_encoder.device)
            
            # Extract the actual embedding tensor
            if isinstance(checkpoint, dict) and "embedding" in checkpoint:
                embedding = checkpoint["embedding"]
            else:
                # Fallback: assume it's a raw tensor
                embedding = checkpoint
            
            # Ensure embedding is on the correct device
            embedding = embedding.to(pipe.text_encoder.device)
            
            # Add token to tokenizer
            num_added = pipe.tokenizer.add_tokens(token)
            if num_added == 0:
                print(f"  Token {token} already exists, skipping...")
                continue
            
            # Resize text encoder token embeddings
            pipe.text_encoder.resize_token_embeddings(len(pipe.tokenizer))
            
            # Get token id and set embedding
            token_id = pipe.tokenizer.convert_tokens_to_ids(token)
            pipe.text_encoder.get_input_embeddings().weight.data[token_id] = embedding
            
            print(f"  Loaded: {token} (id={token_id})")
        
        print(f"Successfully loaded {len(embedding_files)} embeddings")

    def _estimate_hidden_size(self, name: str, unet) -> int:
        """Get the actual hidden size for an attention processor by inspecting UNet structure."""
        # Parse the name to find the actual attention module
        # Example names: "down_blocks.0.attentions.0.transformer_blocks.0.attn1.processor"
        parts = name.split(".")
        
        # Navigate through the UNet to find the actual attention module
        try:
            module = unet
            for part in parts[:-1]:  # Exclude "processor" at the end
                if part.isdigit():
                    module = module[int(part)]
                else:
                    module = getattr(module, part)
            
            # Get the hidden size from the attention module
            if hasattr(module, 'to_q'):
                return module.to_q.in_features
            elif hasattr(module, 'query'):
                return module.query.in_features
        except (AttributeError, IndexError, KeyError):
            pass
        
        # Fallback: use a reasonable default
        return 320

    def _inject_lora(self, pipe: StableDiffusionPipeline):
        attn_procs = {}
        base_cls = LoRAAttnProcessor2_0 if version.parse(torch.__version__) >= version.parse("2.0.0") else LoRAAttnProcessor
        if not inspect.isclass(base_cls):
            base_cls = LoRAAttnProcessor
        ctor_params = inspect.signature(base_cls.__init__).parameters
        accepts_hidden = "hidden_size" in ctor_params
        accepts_cross = "cross_attention_dim" in ctor_params
        rank_key = "rank" if "rank" in ctor_params else ("r" if "r" in ctor_params else None)
        alpha_key = (
            "alpha"
            if "alpha" in ctor_params
            else ("network_alpha" if "network_alpha" in ctor_params else None)
        )

        lora_cls = base_cls
        if inspect.isclass(base_cls) and not issubclass(base_cls, torch.nn.Module):
            class _LoRAWrapper(torch.nn.Module):
                def __init__(self, **kwargs):
                    super().__init__()
                    self.processor = base_cls(**kwargs)
                    self._register_inner_state()

                def _register_inner_state(self) -> None:
                    for name, value in vars(self.processor).items():
                        if isinstance(value, torch.nn.Module):
                            self.add_module(name, value)
                        elif isinstance(value, torch.nn.Parameter):
                            self.register_parameter(name, value)
                        elif isinstance(value, torch.Tensor) and value.requires_grad:
                            self.register_parameter(name, torch.nn.Parameter(value))

                def forward(self, *args, **kwargs):  # pragma: no cover - passthrough
                    return self.processor(*args, **kwargs)

            lora_cls = _LoRAWrapper

        # Get the actual cross_attention_dim from UNet config (it should match text encoder output)
        unet_cross_dim = pipe.unet.config.cross_attention_dim
        print(f"UNet cross_attention_dim from config: {unet_cross_dim}")
        print(f"Text encoder hidden_size: {pipe.text_encoder.config.hidden_size}")

        for name in pipe.unet.attn_processors.keys():
            # For cross-attention (attn2), use UNet's configured cross_attention_dim
            # For self-attention (attn1), cross_attention_dim is None
            if name.endswith("attn2.processor"):
                cross_attention_dim = unet_cross_dim if isinstance(unet_cross_dim, int) else unet_cross_dim
            else:
                cross_attention_dim = None
            hidden_size = self._estimate_hidden_size(name, pipe.unet)
            kwargs = {}
            if rank_key:
                kwargs[rank_key] = self.cfg.rank
            if alpha_key:
                kwargs[alpha_key] = self.cfg.alpha
            if accepts_hidden:
                kwargs["hidden_size"] = hidden_size
            if accepts_cross and cross_attention_dim is not None:
                kwargs["cross_attention_dim"] = cross_attention_dim
            attn_procs[name] = lora_cls(**kwargs)
        pipe.unet.set_attn_processor(attn_procs)
        lora_layers = AttnProcsLayers(pipe.unet.attn_processors)
        lora_layers.to(pipe.unet.device)
        lora_layers.requires_grad_(True)
        return lora_layers

    @staticmethod 
    def _export_attn_procs(unet, output_path: Path) -> None:
        layers = AttnProcsLayers(unet.attn_processors).to(unet.device)
        state = {}
        for key, tensor in layers.state_dict().items():
            if isinstance(tensor, torch.Tensor):
                state[key] = tensor.detach().to("cpu")
            else:
                state[key] = tensor
        save_file(state, str(output_path), metadata={"format": "pt"})

    @staticmethod
    def _collate_fn(batch: List[LoRASample]) -> Dict[str, List]:
        pixel_values = torch.stack([sample.pixel_values for sample in batch])
        prompts = [sample.prompt for sample in batch]
        return {"pixel_values": pixel_values, "prompts": prompts}

    @staticmethod
    def _encode_images(pipe: StableDiffusionPipeline, pixel_values: torch.Tensor) -> torch.Tensor:
        pixel_values = pixel_values.to(device=pipe.vae.device, dtype=pipe.vae.dtype)
        with torch.no_grad():
            latents = pipe.vae.encode(pixel_values).latent_dist.sample()
        return latents * 0.18215

    def train(self, splits: DatasetSplits, token_map: Dict[str, str], output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Look for textual inversion embeddings in outputs/textual_inversion
        embeddings_dir = output_dir.parent / "textual_inversion"
        pipe = self.prepare_pipeline(token_embeddings_dir=embeddings_dir)
        lora_layers = self._inject_lora(pipe)

        trainable_params = [p for p in lora_layers.parameters() if p.requires_grad]
        if not trainable_params:
            raise RuntimeError(
                "LoRA attention processors have no trainable parameters. "
                "Please upgrade diffusers to a version that supports LoRA or adjust the trainer."
            )

        dataset = LoRADataset(splits.train, token_map, self.cfg.prompt_template, self.cfg.resolution)
        dataloader = DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            collate_fn=self._collate_fn,
        )

        optimizer = torch.optim.AdamW(trainable_params, lr=self.cfg.learning_rate)
        noise_scheduler = DDPMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False)
        lr_scheduler = get_scheduler(
            self.cfg.lr_scheduler,
            optimizer,
            num_warmup_steps=self.cfg.lr_warmup_steps,
            num_training_steps=self.cfg.steps,
        )

        tokenizer = pipe.tokenizer
        text_encoder = pipe.text_encoder
        device = pipe.unet.device
        grad_accum = self.cfg.gradient_accumulation_steps
        progress_bar = tqdm(range(self.cfg.steps), desc="LoRA training", leave=False)

        optimizer.zero_grad()
        global_step = 0
        running_loss = 0.0
        
        # Initialize metrics tracking
        metrics_history = {
            "steps": [],
            "loss": [],
            "learning_rate": []
        }
        
        while global_step < self.cfg.steps:
            for batch in dataloader:
                pixel_values = batch["pixel_values"].to(device=device, dtype=pipe.unet.dtype)
                prompts = batch["prompts"]
                text_inputs = tokenizer(
                    prompts,
                    padding="max_length",
                    truncation=True,
                    max_length=tokenizer.model_max_length,
                    return_tensors="pt",
                )
                input_ids = text_inputs.input_ids.to(device)
                with torch.no_grad():
                    encoder_hidden_states = text_encoder(input_ids)[0]
                encoder_hidden_states = encoder_hidden_states.to(dtype=pipe.unet.dtype)

                latents = self._encode_images(pipe, pixel_values)
                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                model_pred = pipe.unet(noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states).sample
                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                
                # Scale loss for gradient accumulation
                scaled_loss = loss / grad_accum
                scaled_loss.backward()
                running_loss += loss.item()  # Accumulate unscaled loss

                if (global_step + 1) % grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(lora_layers.parameters(), self.cfg.max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    
                    # Record metrics after optimizer step
                    avg_loss = running_loss / grad_accum
                    current_lr = optimizer.param_groups[0]["lr"]
                    metrics_history["steps"].append(global_step)
                    metrics_history["loss"].append(avg_loss)
                    metrics_history["learning_rate"].append(current_lr)
                    
                    progress_bar.set_postfix({"loss": f"{avg_loss:.4f}", "lr": f"{current_lr:.2e}"})
                    running_loss = 0.0  # Reset after recording

                progress_bar.update(1)
                global_step += 1
                if global_step >= self.cfg.steps:
                    break

        output_path = output_dir / "lora.safetensors"
        attn_state = pipe.unet.attn_processors
        missing = [name for name, proc in attn_state.items() if proc is None]
        if missing:
            raise RuntimeError(
                "LoRA attention processors missing for: "
                + ", ".join(missing)
                + ". Ensure _inject_lora wires every processor before saving."
            )
        self._export_attn_procs(pipe.unet, output_path)
        
        # Save training metrics
        metrics_path = output_dir / "lora_training_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics_history, f, indent=2)
        print(f"Training metrics saved to: {metrics_path}")

        target_modules = sorted(pipe.unet.attn_processors.keys())
        lora_metadata = {
            "model_id": self.cfg.model_id,
            "rank": self.cfg.rank,
            "alpha": self.cfg.alpha,
            "dropout_rate": self.cfg.dropout_rate,
            "target_modules": target_modules,
            "resolution": self.cfg.resolution,
            "seed": self.cfg.seed,
            "prompt_template": self.cfg.prompt_template,
            "mixed_precision": self.cfg.mixed_precision,
            "training_hyperparameters": {
                "learning_rate": self.cfg.learning_rate,
                "steps": self.cfg.steps,
                "batch_size": self.cfg.batch_size,
                "gradient_accumulation_steps": self.cfg.gradient_accumulation_steps,
                "optimizer": "AdamW",
                "lr_scheduler": self.cfg.lr_scheduler,
                "lr_warmup_steps": self.cfg.lr_warmup_steps,
                "max_grad_norm": self.cfg.max_grad_norm,
            },
        }

        config_path = output_dir / "lora_config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(lora_metadata, f, indent=2, ensure_ascii=False)
        print(f"LoRA config saved to: {config_path}")
        
        pipe.to("cpu")
        progress_bar.close()
        return output_path

try:  # diffusers >=0.17
    from diffusers.models.attention_processor import AttnProcsLayers, LoRAAttnProcessor, LoRAAttnProcessor2_0
except ImportError:  # fallback for older diffusers
    from diffusers.models.attention_processor import LoRAAttnProcessor, LoRAAttnProcessor2_0  # type: ignore

    class AttnProcsLayers(torch.nn.Module):  # minimal shim that exposes LoRA parameters
        def __init__(self, attn_processors):
            super().__init__()
            for idx, (name, processor) in enumerate(attn_processors.items()):
                safe_name = name.replace(".", "_")
                if hasattr(self, safe_name):
                    safe_name = f"{safe_name}_{idx}"
                self.add_module(safe_name, processor)

        def forward(self, *args, **kwargs):  # pragma: no cover - not used
            raise RuntimeError("AttnProcsLayers shim is a container only")


# Provide a module-level heuristic for hidden size used by attention processors
def _estimate_hidden_size(name: str, block_out_channels: List[int]) -> int:
    # Names typically include down_blocks.k.attentions.l ... or mid_block/up_blocks
    if name.startswith("mid_block"):
        return block_out_channels[len(block_out_channels) // 2]
    if name.startswith("up_blocks"):
        return block_out_channels[-1]
    # default to first block
    return block_out_channels[0]


def get_lora_processor_class():
    """
    Return a LoRA attention processor class compatible with the current environment.
    If diffusers provides torch.nn.Module subclasses, return them directly; otherwise,
    return the fallback implementations defined in this module.
    """
    # Prefer 2_0 when torch >= 2.0
    preferred = LoRAAttnProcessor2_0 if version.parse(torch.__version__) >= version.parse("2.0.0") else LoRAAttnProcessor
    # If the chosen class is not a torch.nn.Module subclass (older diffusers), use our fallback
    try:
        if inspect.isclass(preferred) and issubclass(preferred, torch.nn.Module):
            return preferred
    except TypeError:
        pass
    # Fallback classes declared below are torch.nn.Module subclasses
    return LoRAAttnProcessor2_0 if 'LoRAAttnProcessor2_0' in globals() else LoRAAttnProcessor


def _needs_lora_fallback(cls: object) -> bool:
    try:
        return not issubclass(cls, torch.nn.Module)
    except TypeError:
        return True


if _needs_lora_fallback(LoRAAttnProcessor) or _needs_lora_fallback(LoRAAttnProcessor2_0):
    class _LoRALinear(torch.nn.Module):
        def __init__(self, in_dim: int, out_dim: int, rank: int, alpha: int):
            super().__init__()
            if rank <= 0:
                raise ValueError("LoRA rank must be positive")
            self.down = torch.nn.Linear(in_dim, rank, bias=False)
            self.up = torch.nn.Linear(rank, out_dim, bias=False)
            self.scale = alpha / rank
            torch.nn.init.normal_(self.down.weight, std=1.0 / rank)
            torch.nn.init.zeros_(self.up.weight)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            dtype = x.dtype
            projected = self.up(self.down(x.to(self.down.weight.dtype))) * self.scale
            return projected.to(dtype)

    class _BaseFallbackLoRA(torch.nn.Module):
        def __init__(self, hidden_size: int, cross_attention_dim: Optional[int] = None, rank: int = 4, alpha: Optional[int] = None):
            super().__init__()
            cross_attention_dim = cross_attention_dim or hidden_size
            alpha = alpha or rank
            self.hidden_size = hidden_size
            self.cross_attention_dim = cross_attention_dim
            self.rank = rank
            self.alpha = alpha
            self.lora_q = _LoRALinear(hidden_size, hidden_size, rank, alpha)
            self.lora_k = _LoRALinear(cross_attention_dim, hidden_size, rank, alpha)
            self.lora_v = _LoRALinear(cross_attention_dim, hidden_size, rank, alpha)
            self.lora_out = _LoRALinear(hidden_size, hidden_size, rank, alpha)

        @staticmethod
        def _reshape_if_needed(hidden_states: torch.Tensor, info: Tuple[int, Optional[int], Optional[int], Optional[int]]) -> torch.Tensor:
            batch_size, channel, height, width = info
            if channel is None:
                return hidden_states
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
            return hidden_states

        def _preprocess(
            self,
            attn,
            hidden_states: torch.Tensor,
            encoder_hidden_states: Optional[torch.Tensor],
            attention_mask: Optional[torch.Tensor],
            temb: Optional[torch.Tensor],
        ):
            residual = hidden_states
            if attn.spatial_norm is not None:
                hidden_states = attn.spatial_norm(hidden_states, temb)
            input_ndim = hidden_states.ndim
            if input_ndim == 4:
                batch_size, channel, height, width = hidden_states.shape
                hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
            else:
                batch_size = hidden_states.shape[0]
                channel = height = width = None
            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
                target_len = hidden_states.shape[1]
            else:
                target_len = encoder_hidden_states.shape[1]
                if attn.norm_cross:
                    encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
            attention_mask = attn.prepare_attention_mask(attention_mask, target_len, batch_size)
            if attn.group_norm is not None:
                hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
            return residual, hidden_states, encoder_hidden_states, attention_mask, input_ndim, (batch_size, channel, height, width)

        def _apply_out(self, attn, hidden_states: torch.Tensor) -> torch.Tensor:
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = hidden_states + self.lora_out(hidden_states)
            hidden_states = attn.to_out[1](hidden_states)
            return hidden_states

    class LoRAAttnProcessor(_BaseFallbackLoRA):
        def __call__(
            self,
            attn,
            hidden_states: torch.Tensor,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            temb: Optional[torch.Tensor] = None,
            *args,
            **kwargs,
        ) -> torch.Tensor:
            residual, hidden_states, encoder_hidden_states, attention_mask, input_ndim, shape_info = self._preprocess(
                attn, hidden_states, encoder_hidden_states, attention_mask, temb
            )
            query = attn.to_q(hidden_states) + self.lora_q(hidden_states)
            key = attn.to_k(encoder_hidden_states) + self.lora_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states) + self.lora_v(encoder_hidden_states)
            query = attn.head_to_batch_dim(query)
            key = attn.head_to_batch_dim(key)
            value = attn.head_to_batch_dim(value)
            attention_probs = attn.get_attention_scores(query, key, attention_mask)
            hidden_states = torch.bmm(attention_probs, value)
            hidden_states = attn.batch_to_head_dim(hidden_states)
            hidden_states = self._apply_out(attn, hidden_states)
            if input_ndim == 4:
                hidden_states = self._reshape_if_needed(hidden_states, shape_info)
            if attn.residual_connection:
                hidden_states = hidden_states + residual
            hidden_states = hidden_states / attn.rescale_output_factor
            return hidden_states

    class LoRAAttnProcessor2_0(_BaseFallbackLoRA):
        def __call__(
            self,
            attn,
            hidden_states: torch.Tensor,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            temb: Optional[torch.Tensor] = None,
            *args,
            **kwargs,
        ) -> torch.Tensor:
            residual, hidden_states, encoder_hidden_states, attention_mask, input_ndim, shape_info = self._preprocess(
                attn, hidden_states, encoder_hidden_states, attention_mask, temb
            )
            query = attn.to_q(hidden_states) + self.lora_q(hidden_states)
            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            key = attn.to_k(encoder_hidden_states) + self.lora_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states) + self.lora_v(encoder_hidden_states)
            batch_size = hidden_states.shape[0]
            inner_dim = key.shape[-1]
            head_dim = inner_dim // attn.heads
            query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            if attention_mask is not None:
                attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])
            if hasattr(F, "scaled_dot_product_attention"):
                hidden_states = F.scaled_dot_product_attention(
                    query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
                )
            else:  # fallback to standard attention
                attention_mask = attention_mask.view(batch_size * attn.heads, 1, -1) if attention_mask is not None else None
                query = query.reshape(batch_size * attn.heads, -1, head_dim)
                key = key.reshape(batch_size * attn.heads, -1, head_dim)
                value = value.reshape(batch_size * attn.heads, -1, head_dim)
                attention_probs = attn.get_attention_scores(query, key, attention_mask)
                hidden_states = torch.bmm(attention_probs, value)
                hidden_states = hidden_states.view(batch_size, attn.heads, -1, head_dim)
            hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
            hidden_states = self._apply_out(attn, hidden_states)
            if input_ndim == 4:
                hidden_states = self._reshape_if_needed(hidden_states, shape_info)
            if attn.residual_connection:
                hidden_states = hidden_states + residual
            hidden_states = hidden_states / attn.rescale_output_factor
            return hidden_states

