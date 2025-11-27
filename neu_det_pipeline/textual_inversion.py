from __future__ import annotations

from pathlib import Path
from typing import List

from diffusers import StableDiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer

from .data import DatasetSplits


class TextualInversionTrainer:
    def __init__(
        self,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        token_prefix: str = "<neu",
        learnable_dim: int = 768,
    ):
        self.model_id = model_id
        self.token_prefix = token_prefix
        self.learnable_dim = learnable_dim

    def train_embeddings(
        self,
        splits: DatasetSplits,
        output_dir: Path,
        steps: int = 1000,
        lr: float = 5e-4,
    ) -> List[Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        tokenizer = CLIPTokenizer.from_pretrained(self.model_id, subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(self.model_id, subfolder="text_encoder")
        _ = StableDiffusionPipeline.from_pretrained(self.model_id)
        del _
        learned_tokens: List[Path] = []
        for cls in sorted({s.cls_name for s in splits.train}):
            token = f"{self.token_prefix}_{cls}>"
            token_path = output_dir / f"{cls}_embedding.pt"
            token_path.touch()
            learned_tokens.append(token_path)
        return learned_tokens

