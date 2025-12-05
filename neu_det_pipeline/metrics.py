from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import cv2
import numpy as np
import torch
from PIL import Image
from rich.console import Console
from skimage.metrics import structural_similarity as ssim
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchvision import transforms

console = Console()


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _stem(path: Path) -> str:
    return path.stem


@dataclass
class DistributionStats:
    mean: float
    median: float
    std: float
    minimum: float
    maximum: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "mean": self.mean,
            "median": self.median,
            "std": self.std,
            "min": self.minimum,
            "max": self.maximum,
        }


@dataclass
class MetricResult:
    fid: Optional[float]
    kid_mean: Optional[float]
    kid_std: Optional[float]
    lpips: Optional[float]
    edge_ssim: Optional[float]
    pairs_used: int
    lpips_distribution: Optional[DistributionStats] = None
    edge_distribution: Optional[DistributionStats] = None
    missing_references: List[str] = field(default_factory=list)
    failed_samples: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "fid": self.fid,
            "kid_mean": self.kid_mean,
            "kid_std": self.kid_std,
            "lpips": self.lpips,
            "edge_ssim": self.edge_ssim,
            "pairs_used": self.pairs_used,
            "lpips_distribution": self.lpips_distribution.to_dict() if self.lpips_distribution else None,
            "edge_distribution": self.edge_distribution.to_dict() if self.edge_distribution else None,
            "missing_references": self.missing_references,
            "failed_samples": self.failed_samples,
        }


class GenerationMetricsEvaluator:
    """Compute distribution-level and structure-level metrics for generated images."""

    def __init__(
        self,
        *,
        device: Optional[str] = None,
        image_size: int = 299,
    ) -> None:
        self.device = device or _default_device()
        self.image_size = image_size
        self.transform = transforms.Compose(
            [
                transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
            ]
        )
        self.fid = FrechetInceptionDistance(normalize=True).to(self.device)
        # Use small subset size to keep memory manageable on constrained GPUs
        self.kid = KernelInceptionDistance(subset_size=50, normalize=True).to(self.device)
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type="vgg").to(self.device)

    def _load_tensor(self, path: Path) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        tensor = self.transform(img).unsqueeze(0).to(self.device)
        return tensor

    def _edge_ssim(self, generated_path: Path, reference_canny: Path) -> Optional[float]:
        gen = cv2.imread(str(generated_path), cv2.IMREAD_GRAYSCALE)
        ref = cv2.imread(str(reference_canny), cv2.IMREAD_GRAYSCALE)
        if gen is None or ref is None:
            return None
        gen_canny = cv2.Canny(gen, 60, 160)
        ref_resized = cv2.resize(ref, (gen_canny.shape[1], gen_canny.shape[0]), interpolation=cv2.INTER_AREA)
        gen_float = gen_canny.astype(np.float32) / 255.0
        ref_float = ref_resized.astype(np.float32) / 255.0
        return float(ssim(ref_float, gen_float, data_range=1.0))

    def evaluate(
        self,
        generated_paths: Sequence[Path],
        reference_map: Dict[str, Path],
        guidance_map: Optional[Dict[str, Dict[str, Path]]] = None,
    ) -> MetricResult:
        lpips_scores: List[float] = []
        edge_scores: List[float] = []
        pairs_used = 0
        missing_refs: List[str] = []
        failed_samples: List[str] = []

        for gen_path in generated_paths:
            stem = _stem(gen_path)
            ref_path = reference_map.get(stem)
            if ref_path is None or not ref_path.exists():
                missing_refs.append(stem)
                continue
            try:
                real_tensor = self._load_tensor(ref_path)
                fake_tensor = self._load_tensor(gen_path)
            except Exception as exc:  # noqa: BLE001
                console.print(f"[yellow]跳过 {gen_path}，无法载入图像: {exc}[/yellow]")
                failed_samples.append(stem)
                continue
            self.fid.update(real_tensor, real=True)
            self.fid.update(fake_tensor, real=False)
            self.kid.update(real_tensor, real=True)
            self.kid.update(fake_tensor, real=False)
            lpips_scores.append(float(self.lpips(fake_tensor, real_tensor).detach().cpu()))
            pairs_used += 1

            if guidance_map:
                canny_path = guidance_map.get(stem, {}).get("canny")
                if canny_path and canny_path.exists():
                    edge_value = self._edge_ssim(gen_path, canny_path)
                    if edge_value is not None:
                        edge_scores.append(edge_value)

        if pairs_used == 0:
            console.print("[yellow]无法评估生成质量：没有可比较的图像对。[/yellow]")
            return MetricResult(None, None, None, None, None, pairs_used)

        fid_value = float(self.fid.compute().cpu())
        kid_mean_value: Optional[float]
        kid_std_value: Optional[float]
        try:
            kid_mean, kid_std = self.kid.compute()
            kid_mean_value = float(kid_mean.cpu())
            kid_std_value = float(kid_std.cpu())
        except ValueError as exc:
            console.print(
                f"[yellow]无法计算 KID（样本不足或配置不匹配）: {exc}. 返回 None。[/yellow]"
            )
            kid_mean_value = None
            kid_std_value = None

        lpips_stats = _summary_stats(lpips_scores)
        edge_stats = _summary_stats(edge_scores)
        result = MetricResult(
            fid=fid_value,
            kid_mean=kid_mean_value,
            kid_std=kid_std_value,
            lpips=float(np.mean(lpips_scores)) if lpips_scores else None,
            edge_ssim=float(np.mean(edge_scores)) if edge_scores else None,
            pairs_used=pairs_used,
            lpips_distribution=lpips_stats,
            edge_distribution=edge_stats,
            missing_references=missing_refs,
            failed_samples=failed_samples,
        )

        # Reset states so the evaluator can be reused
        self.fid.reset()
        self.kid.reset()
        self.lpips.reset()
        return result


def _summary_stats(values: Sequence[float]) -> Optional[DistributionStats]:
    if not values:
        return None
    arr = np.array(values, dtype=np.float32)
    return DistributionStats(
        mean=float(np.mean(arr)),
        median=float(np.median(arr)),
        std=float(np.std(arr)),
        minimum=float(np.min(arr)),
        maximum=float(np.max(arr)),
    )


def save_metrics(result: MetricResult, dest: Path, *, extra: Optional[Dict[str, object]] = None) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as f:
        payload = {
            "metrics": result.to_dict(),
            "run": extra or {},
        }
        json.dump(payload, f, indent=2)
    console.print(f"[green]生成质量指标已写入 {dest}[/green]")
