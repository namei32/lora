from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, List, Set

import cv2
import numpy as np
from controlnet_aux import HEDdetector, MidasDetector
from rich.progress import track

from .data import DefectSample


class GuidanceExtractor:
    def __init__(self, hed_repo_id: str, midas_repo_id: str, hed_ckpt: str, midas_ckpt: str):
        self.hed = HEDdetector.from_pretrained(hed_repo_id, filename=hed_ckpt)
        self.midas = MidasDetector.from_pretrained(midas_repo_id, filename=midas_ckpt)

    def compute_guidance(self, sample: DefectSample) -> Tuple[np.ndarray, np.ndarray]:
        rgb = cv2.cvtColor(cv2.imread(str(sample.image_path)), cv2.COLOR_BGR2RGB)
        hed_map_pil = self.hed(rgb)
        depth_map = self.midas(rgb)
        hed_map = np.array(hed_map_pil)
        if depth_map.dtype != np.uint8:
            depth_min, depth_max = depth_map.min(), depth_map.max()
            denom = max(depth_max - depth_min, 1e-6)
            depth_map = ((depth_map - depth_min) / denom * 255).astype(np.uint8)
        if hed_map.dtype != np.uint8:
            hed_map = hed_map.astype(np.uint8)
        return hed_map, depth_map

    def batch_process(self, samples: list[DefectSample], out_dir: Path) -> Dict[str, Dict[str, Path]]:
        out_dir.mkdir(parents=True, exist_ok=True)
        outputs: Dict[str, Dict[str, Path]] = {}
        for sample in track(samples, description="Extracting HED/MiDaS"):
            hed_map, depth_map = self.compute_guidance(sample)
            hed_path = out_dir / f"{sample.image_path.stem}_hed.png"
            depth_path = out_dir / f"{sample.image_path.stem}_depth.png"
            cv2.imwrite(str(hed_path), hed_map)
            cv2.imwrite(str(depth_path), depth_map)
            outputs[sample.image_path.stem] = {"hed": hed_path, "depth": depth_path}
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
