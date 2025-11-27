from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
from rich.table import Table
from rich.console import Console

from .data import DefectSample

console = Console()


def compute_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    x_min = max(box_a[0], box_b[0])
    y_min = max(box_a[1], box_b[1])
    x_max = min(box_a[2], box_b[2])
    y_max = min(box_a[3], box_b[3])
    inter = max(0, x_max - x_min) * max(0, y_max - y_min)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def validate_alignment(
    originals: List[DefectSample],
    generated_paths: List[Path],
    annotations: Dict[str, np.ndarray],
    iou_threshold: float = 0.5,
) -> Table:
    table = Table(title="Alignment Report")
    table.add_column("Sample")
    table.add_column("IoU", justify="right")
    table.add_column("Pass", justify="center")
    for sample, gen_path in zip(originals, generated_paths):
        gen_img = cv2.imread(str(gen_path))
        h_ratio = gen_img.shape[0] / 200
        w_ratio = gen_img.shape[1] / 200
        ref_box = annotations[sample.image_path.stem]
        scaled_box = np.array(
            [
                ref_box[0] * w_ratio,
                ref_box[1] * h_ratio,
                ref_box[2] * w_ratio,
                ref_box[3] * h_ratio,
            ]
        )
        iou = compute_iou(ref_box, scaled_box)
        table.add_row(sample.image_path.name, f"{iou:.2f}", "✅" if iou >= iou_threshold else "⚠️")
    console.print(table)
    return table

