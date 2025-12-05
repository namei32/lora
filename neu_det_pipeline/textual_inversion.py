from __future__ import annotations

import json
from pathlib import Path
from typing import List
import warnings

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from diffusers import DDPMScheduler, StableDiffusionPipeline
from diffusers.optimization import get_scheduler
from transformers import CLIPTokenizer, CLIPTextModel
from tqdm import tqdm
import numpy as np
import cv2

from .data import DatasetSplits, DefectSample, load_image

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class TextualInversionDataset(Dataset):
    """
    Dataset for textual inversion training on a specific defect class.
    
    Component 1 of the workflow (CLIP):
    Employs the textual inversion feature of the Contrastive Language-Image 
    Pre-training (CLIP) model to generate prompt keywords corresponding to 
    each category of the original images.
    
    Returns normalized image tensors in [0, 1] range.
    """

    def __init__(self, samples: List[DefectSample], cls_name: str, resolution: int = 512):
        self.samples = [s for s in samples if s.cls_name == cls_name]
        self.resolution = resolution

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> torch.Tensor:
        sample = self.samples[idx]
        image = load_image(sample.image_path)
        
        # Center crop to square
        h, w = image.shape[:2]
        size = min(h, w)
        top = (h - size) // 2
        left = (w - size) // 2
        image = image[top:top + size, left:left + size]
        
        # Resize to target resolution
        image = cv2.resize(image, (self.resolution, self.resolution), interpolation=cv2.INTER_AREA)
        
        # Normalize to [0, 1] and convert to tensor
        image = image.astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image).permute(2, 0, 1)
        
        return image_tensor


class TextualInversionTrainer:
    """
    Textual Inversion Trainer implementing the first component of the workflow.
    
    This module employs the textual inversion feature of the Contrastive Language-Image 
    Pre-training (CLIP) model to generate prompt keywords corresponding to each category 
    of the original images.
    
    By learning category-specific embeddings, the model generates high-fidelity prompts 
    that accurately represent different types of steel defects.
    """

    def __init__(
        self,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        token_prefix: str = "<neu",
        initializer_token: str = "steel",
    ):
        self.model_id = model_id
        self.token_prefix = token_prefix
        self.initializer_token = initializer_token

    def train_embeddings(
        self,
        splits: DatasetSplits,
        output_dir: Path,
        steps: int = 800,
        lr: float = 5e-4,
        batch_size: int = 4,
        resolution: int = 512,
        prompt_template: str = "macro photo of {token} steel defect",
    ) -> List[Path]:
        """
        Train CLIP embeddings for each defect category using textual inversion.
        
        Args:
            splits: Training/validation data splits
            output_dir: Directory to save learned embeddings
            steps: Number of training steps per class
            lr: Learning rate
            batch_size: Batch size for training
            resolution: Image resolution
            prompt_template: Template for generating prompts
            
        Returns:
            List of paths to saved embedding files
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Load models - force float32 to avoid dtype issues
        print("Loading Stable Diffusion pipeline...")
        tokenizer = CLIPTokenizer.from_pretrained(self.model_id, subfolder="tokenizer")
        
        # Load each component separately with explicit float32 dtype
        # This prevents HuggingFace from using cached float16 variants
        print("Loading text encoder...")
        text_encoder = CLIPTextModel.from_pretrained(
            self.model_id, 
            subfolder="text_encoder",
            torch_dtype=torch.float32,
            variant=None  # Disable variant loading to avoid fp16
        ).to(device)
        
        print("Loading VAE and UNet...")
        pipe = StableDiffusionPipeline.from_pretrained(
            self.model_id,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            torch_dtype=torch.float32,
            variant=None,  # Disable variant loading
            safety_checker=None,
            requires_safety_checker=False,
        ).to(device)
        
        # Get references
        text_encoder = pipe.text_encoder
        vae = pipe.vae
        unet = pipe.unet
        
        # Verify dtypes
        print(f"Text encoder dtype: {next(text_encoder.parameters()).dtype}")
        print(f"VAE dtype: {next(vae.parameters()).dtype}")
        print(f"UNet dtype: {next(unet.parameters()).dtype}")
        
        # Freeze all parameters except embeddings
        vae.requires_grad_(False)
        unet.requires_grad_(False)
        text_encoder.requires_grad_(False)
        
        # Load noise scheduler
        noise_scheduler = DDPMScheduler.from_pretrained(self.model_id, subfolder="scheduler")
        
        learned_embeddings: List[Path] = []
        
        # Get unique classes
        classes = sorted(set(s.cls_name for s in splits.train))
        print(f"Training embeddings for {len(classes)} classes: {classes}")
        
        # Train embedding for each class
        for cls_name in classes:
            print(f"\n{'='*60}")
            print(f"Training embedding for class: {cls_name}")
            print(f"{'='*60}")
            
            # Create dataset for this class
            dataset = TextualInversionDataset(splits.train, cls_name, resolution)
            if len(dataset) == 0:
                print(f"Warning: No samples found for class {cls_name}, skipping...")
                continue
            
            print(f"Dataset size: {len(dataset)} images")
            
            # Define placeholder token
            placeholder_token = f"{self.token_prefix}_{cls_name}>"
            
            # Train this token
            embedding_path = self._train_single_embedding(
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                vae=vae,
                unet=unet,
                noise_scheduler=noise_scheduler,
                dataset=dataset,
                placeholder_token=placeholder_token,
                prompt_template=prompt_template,
                steps=steps,
                lr=lr,
                batch_size=batch_size,
                device=device,
                output_dir=output_dir,
            )
            
            learned_embeddings.append(embedding_path)
        
        # Cleanup
        print("\nCleaning up...")
        del pipe, text_encoder, vae, unet
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print(f"\n✓ Textual inversion complete! Saved {len(learned_embeddings)} embeddings.")
        return learned_embeddings

    def _train_single_embedding(
        self,
        tokenizer,
        text_encoder,
        vae,
        unet,
        noise_scheduler,
        dataset,
        placeholder_token: str,
        prompt_template: str,
        steps: int,
        lr: float,
        batch_size: int,
        device,
        output_dir: Path,
    ) -> Path:
        """Train a single embedding for one defect class."""
        
        # Add placeholder token to tokenizer
        num_added_tokens = tokenizer.add_tokens(placeholder_token)
        token_id = tokenizer.convert_tokens_to_ids(placeholder_token)
        
        if num_added_tokens == 0:
            # Token already exists
            print(f"Token {placeholder_token} already exists with ID {token_id}")
        else:
            # Resize token embeddings
            text_encoder.resize_token_embeddings(len(tokenizer))
            
            # CRITICAL: After resizing, convert back to float32
            # resize_token_embeddings may create new layers in default dtype
            text_encoder = text_encoder.to(torch.float32)
            
            # Verify dtype
            print(f"After resize - Text encoder dtype: {next(text_encoder.parameters()).dtype}")
            
            # Initialize from initializer token
            embedding_layer = text_encoder.get_input_embeddings()
            
            # Get initializer token ID
            initializer_tokens = tokenizer.encode(self.initializer_token, add_special_tokens=False)
            if len(initializer_tokens) == 0:
                raise ValueError(f"Initializer token '{self.initializer_token}' not found")
            initializer_token_id = initializer_tokens[0]
            
            # Initialize new token with initializer token's embedding
            with torch.no_grad():
                embedding_layer.weight[token_id] = embedding_layer.weight[initializer_token_id].clone()
            
            print(f"Added new token {placeholder_token} with ID {token_id}")
        
        # Get embedding layer
        embedding_layer = text_encoder.get_input_embeddings()
        
        # Store original embeddings for restoration
        original_embeddings = embedding_layer.weight.data.clone()
        
        # Only optimize the new token embedding
        embedding_layer.weight.requires_grad_(True)
        
        # Create dataloader
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=0,
            pin_memory=True
        )
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(
            [embedding_layer.weight],
            lr=lr,
            betas=(0.9, 0.999),
            weight_decay=1e-2,
            eps=1e-08,
        )
        
        # Setup learning rate scheduler
        lr_scheduler = get_scheduler(
            "cosine",
            optimizer=optimizer,
            num_warmup_steps=min(100, steps // 10),
            num_training_steps=steps,
        )
        
        # Training loop
        progress_bar = tqdm(range(steps), desc=f"Training {placeholder_token}")
        
        global_step = 0
        text_encoder.train()
        
        # Initialize metrics tracking
        metrics_history = {
            "token": placeholder_token,
            "steps": [],
            "loss": [],
            "learning_rate": []
        }
        
        while global_step < steps:
            for batch in dataloader:
                # Move batch to device
                images = batch.to(device, dtype=torch.float32)
                
                # Encode images to latent space
                with torch.no_grad():
                    # Normalize images to [-1, 1]
                    latents = vae.encode(images * 2 - 1).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor
                
                # Sample noise
                noise = torch.randn_like(latents)
                
                # Sample random timesteps
                timesteps = torch.randint(
                    0, 
                    noise_scheduler.config.num_train_timesteps,
                    (latents.shape[0],),
                    device=device,
                    dtype=torch.long
                )
                
                # Add noise to latents
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                # Get text embeddings
                prompt = prompt_template.format(token=placeholder_token)
                text_inputs = tokenizer(
                    [prompt] * images.shape[0],
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                
                # Get encoder hidden states
                encoder_hidden_states = text_encoder(text_inputs.input_ids.to(device))[0]
                
                # Predict the noise residual
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                ).sample
                
                # Calculate loss
                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                
                # Backpropagate
                loss.backward()
                
                # Zero out gradients for all tokens except the placeholder
                if embedding_layer.weight.grad is not None:
                    # Create mask for all tokens except the placeholder
                    grad_mask = torch.arange(len(tokenizer), device=device) != token_id
                    embedding_layer.weight.grad[grad_mask] = 0.0
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(embedding_layer.parameters(), max_norm=1.0)
                
                # Update weights
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
                # Restore original embeddings for all tokens except the placeholder
                with torch.no_grad():
                    mask = torch.arange(len(tokenizer), device=device) != token_id
                    embedding_layer.weight[mask] = original_embeddings[mask].to(device)
                
                # Record metrics
                current_lr = optimizer.param_groups[0]["lr"]
                metrics_history["steps"].append(global_step)
                metrics_history["loss"].append(loss.item())
                metrics_history["learning_rate"].append(current_lr)
                
                # Update progress
                progress_bar.update(1)
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{current_lr:.2e}"})
                
                global_step += 1
                if global_step >= steps:
                    break
        
        progress_bar.close()
        
        # Save learned embedding
        learned_embeds = embedding_layer.weight[token_id].detach().cpu()
        
        save_path = output_dir / f"{placeholder_token.strip('<>')}_embedding.pt"
        torch.save(
            {
                "token": placeholder_token,
                "embedding": learned_embeds,
                "token_id": token_id,
            },
            save_path
        )
        
        print(f"Saved embedding to: {save_path}")
        
        # Save training metrics
        metrics_path = output_dir / f"{placeholder_token.strip('<>')}_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics_history, f, indent=2)
        print(f"Training metrics saved to: {metrics_path}")
        
        # Reset gradients
        embedding_layer.weight.requires_grad_(False)
        
        return save_path
