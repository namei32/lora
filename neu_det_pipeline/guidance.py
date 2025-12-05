from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, List, Set

import cv2
import numpy as np
from controlnet_aux import HEDdetector, MidasDetector
from rich.progress import track

from .data import DefectSample


class GuidanceExtractor:
    """
    Component 3 of the workflow (Feature Extraction):
    
    Leverages HED (Holistically-nested Edge Detection) and MiDaS (depth map 
    estimation) models to obtain approximate contours and depth maps of images.
    
    Extracts three types of guidance features:
    1. Canny edge from original input image (original input features)
    2. HED contours (approximate contours, contour features)
    3. MiDaS depth maps (depth map features)
    
    These features are used by ControlNet for controllable generation.
    """
    def __init__(self, hed_repo_id: str, midas_repo_id: str, hed_ckpt: str, midas_ckpt: str):
        self.hed = HEDdetector.from_pretrained(hed_repo_id, filename=hed_ckpt)
        self.midas = MidasDetector.from_pretrained(midas_repo_id, filename=midas_ckpt)

    def compute_guidance(self, sample: DefectSample) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute three guidance features:
        - Canny edge: approximate contours from original image
        - HED map: holistically-nested edge detection
        - Depth map: MiDaS depth estimation
        """
        rgb = cv2.cvtColor(cv2.imread(str(sample.image_path)), cv2.COLOR_BGR2RGB)
        
        # 1. Extract Canny edge from original input for structure preservation
        # Lower thresholds to reduce high-frequency noise
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        # Apply slight Gaussian blur to reduce noise before edge detection
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        canny_map = cv2.Canny(gray, threshold1=60, threshold2=160)
        
        # 2. Extract HED contours for detailed edge features
        hed_map_pil = self.hed(rgb)
        hed_map = np.array(hed_map_pil)
        if hed_map.dtype != np.uint8:
            hed_map = hed_map.astype(np.uint8)
        
        # 3. Extract MiDaS depth map for spatial structure
        depth_map = self.midas(rgb)
        if depth_map.dtype != np.uint8:
            depth_min, depth_max = depth_map.min(), depth_map.max()
            denom = max(depth_max - depth_min, 1e-6)
            depth_map = ((depth_map - depth_min) / denom * 255).astype(np.uint8)
        
        return canny_map, hed_map, depth_map

    def batch_process(self, samples: list[DefectSample], out_dir: Path) -> Dict[str, Dict[str, Path]]:
        """
        Process all samples and extract three guidance features:
        original input features (canny), contour features (HED), and depth map features (MiDaS)
        """
        out_dir.mkdir(parents=True, exist_ok=True)
        outputs: Dict[str, Dict[str, Path]] = {}
        for sample in track(samples, description="Extracting Canny/HED/MiDaS"):
            canny_map, hed_map, depth_map = self.compute_guidance(sample)
            canny_path = out_dir / f"{sample.image_path.stem}_canny.png"
            hed_path = out_dir / f"{sample.image_path.stem}_hed.png"
            depth_path = out_dir / f"{sample.image_path.stem}_depth.png"
            cv2.imwrite(str(canny_path), canny_map)
            cv2.imwrite(str(hed_path), hed_map)
            cv2.imwrite(str(depth_path), depth_map)
            outputs[sample.image_path.stem] = {
                "canny": canny_path,
                "hed": hed_path,
                "depth": depth_path
            }
        return outputs


def load_guidance_map(
    samples: List[DefectSample],
    guidance_dir: Path,
    modalities: List[str],
) -> Dict[str, Dict[str, Path]]:
    unique_modalities: Set[str] = set(modalities)
    conditioning: Dict[str, Dict[str, Path]] = {}
    for sample in samples:
        stem = sample.image_path.stem
        entry: Dict[str, Path] = {}
        for modality in unique_modalities:
            guide_path = guidance_dir / f"{stem}_{modality}.png"
            if not guide_path.exists():
                raise FileNotFoundError(
                    f"缺少 {modality} 引导文件: {guide_path}. 请先运行 guidance 命令或检查文件命名。"
                )
            entry[modality] = guide_path
        conditioning[stem] = entry
    return conditioning
