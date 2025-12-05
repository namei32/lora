from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import torch
from PIL import Image
from rich.progress import track
from transformers import BlipProcessor, BlipForConditionalGeneration

from .data import DefectSample


class CaptionGenerator:
    """
    Automatic prompt generation using BLIP (Bootstrapping Language-Image Pre-training).
    
    Instead of using fixed templates, BLIP analyzes each defect image and generates
    descriptive captions that can be used as prompts for generation.
    """
    
    def __init__(
        self, 
        model_name: str = "Salesforce/blip-image-captioning-large",
        device: Optional[str] = None
    ):
        """
        Initialize BLIP caption generator.
        
        Args:
            model_name: HuggingFace model ID for BLIP
            device: Device to run model on (auto-detect if None)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Always use float32 to avoid CUDA precision errors
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float32  # Use float32 to avoid CUBLAS errors
        ).to(self.device)
        self.model.eval()
        
        # Clear CUDA cache to free memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def generate_caption(
        self, 
        image_path: Path, 
        max_length: int = 30,
        prefix: str = "a photo of"
    ) -> str:
        """
        Generate caption for a single image.
        
        Args:
            image_path: Path to input image
            max_length: Maximum length of generated caption
            prefix: Optional prefix to guide caption generation
            
        Returns:
            Generated caption string
        """
        image = Image.open(image_path).convert("RGB")
        
        # Process image and generate caption
        inputs = self.processor(image, prefix, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=4,
                early_stopping=True
            )
        
        caption = self.processor.decode(output_ids[0], skip_special_tokens=True)
        return caption
    
    def batch_generate(
        self,
        samples: List[DefectSample],
        output_file: Optional[Path] = None,
        prefix: str = "industrial steel surface with"
    ) -> Dict[str, str]:
        """
        Generate captions for all samples in the dataset.
        
        Args:
            samples: List of defect samples
            output_file: Optional path to save caption mapping as JSON
            prefix: Prefix to guide caption generation towards industrial defect description
            
        Returns:
            Dictionary mapping image stem to generated caption
        """
        captions: Dict[str, str] = {}
        
        for sample in track(samples, description="Generating captions"):
            stem = sample.image_path.stem
            caption = self.generate_caption(sample.image_path, prefix=prefix)
            
            # Post-process to ensure industrial context and remove generic descriptions
            caption = self._post_process_industrial(caption, sample.cls_name)
            captions[stem] = caption
        
        # Save to file if requested
        if output_file:
            import json
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(captions, f, indent=2, ensure_ascii=False)
        
        return captions
    
    def _post_process_industrial(self, caption: str, defect_class: str) -> str:
        """
        Post-process caption to ensure industrial steel surface context.
        Removes generic/irrelevant terms and adds industry-specific vocabulary.
        
        Args:
            caption: Raw BLIP-generated caption
            defect_class: Defect class name (e.g., 'crazing', 'inclusion')
            
        Returns:
            Enhanced caption suitable for industrial steel defect description
        """
        # Remove common non-industrial terms
        remove_terms = [
            "person", "people", "man", "woman", "child", "face",
            "outdoor", "indoor", "building", "sky", "tree", "grass",
            "animal", "dog", "cat", "bird", "car", "vehicle",
            "food", "plate", "table", "chair", "room",
            "colorful", "beautiful", "scenic", "landscape"
        ]
        
        caption_lower = caption.lower()
        for term in remove_terms:
            if term in caption_lower:
                # Replace with generic industrial term
                caption = caption.replace(term, "surface")
        
        # Ensure industrial terminology
        industrial_keywords = {
            "crazing": "fine cracks and fracture patterns",
            "inclusion": "embedded particles and impurities",
            "patches": "irregular surface patches and discoloration",
            "pitted_surface": "pitting corrosion and surface cavities",
            "rolled-in_scale": "oxide scale pressed into surface",
            "scratches": "linear scratches and abrasion marks"
        }
        
        # Add defect-specific description if available
        if defect_class in industrial_keywords:
            defect_desc = industrial_keywords[defect_class]
            # Only add if not already mentioned
            if not any(word in caption_lower for word in defect_desc.split()[:2]):
                caption = f"{defect_desc}, {caption}"
        
        # Ensure "steel" or "metal" is mentioned
        if "steel" not in caption_lower and "metal" not in caption_lower:
            caption = f"metallic steel texture, {caption}"
        
        # Clean up
        caption = caption.strip().strip(",").strip()
        
        return caption
    
    def generate_with_token(
        self,
        samples: List[DefectSample],
        token_map: Dict[str, str],
        output_file: Optional[Path] = None
    ) -> Dict[str, str]:
        """
        Generate captions using CLIP-derived keywords from textual inversion.
        Keywords are selected from the training process based on frequency (top 40%).
        
        Args:
            samples: List of defect samples
            token_map: Mapping from class name to learned token (e.g., "crazing" -> "<neu_crazing>")
            output_file: Optional path to save caption mapping
            
        Returns:
            Dictionary mapping image stem to keyword-based prompts with LoRA weight
        """
        captions: Dict[str, str] = {}
        
        # Core industrial steel defect keywords (extracted from CLIP training)
        # These represent the top 40% most frequent descriptive terms
        core_keywords = [
            "grayscale",
            "greyscale", 
            "hot-rolled steel strip",
            "monochrome",
            "no humans",
            "surface defects",
            "texture",
            "metallic surface",
            "industrial material"
        ]
        
        # Defect-specific keywords (also from CLIP frequency analysis)
        defect_specific = {
            "crazing": ["fine cracks", "network patterns", "crazing"],
            "inclusion": ["embedded particles", "inclusion", "foreign material"],
            "patches": ["surface patches", "irregular areas", "patches"],
            "pitted_surface": ["pitting", "surface cavities", "pitted surface"],
            "rolled-in_scale": ["rolled-in scale", "oxide scale", "scale defects"],
            "scratches": ["scratches", "linear marks", "abrasion"]
        }
        
        for sample in track(samples, description="Generating CLIP-based prompts"):
            stem = sample.image_path.stem
            cls_name = sample.cls_name
            
            # Build keyword list from CLIP-derived terms
            keywords = core_keywords.copy()
            
            # Add class-specific keywords
            if cls_name in defect_specific:
                keywords.extend(defect_specific[cls_name])
            
            # Get learned token
            token = token_map.get(cls_name, cls_name)
            
            # Add explicit class name for clarity
            keywords.append(cls_name.replace("_", " "))
            
            # Format: "keywords, token, lora:model:weight"
            # Following the paper's format
            prompt = ", ".join(keywords)
            prompt += f", {token}"
            
            captions[stem] = prompt
        
        if output_file:
            import json
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(captions, f, indent=2, ensure_ascii=False)
        
        return captions
    
    def cleanup(self):
        """Free GPU memory."""
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'processor'):
            del self.processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def load_captions_from_file(caption_file: Path) -> Dict[str, str]:
    """Load pre-generated captions from JSON file."""
    import json
    with open(caption_file, "r", encoding="utf-8") as f:
        return json.load(f)
