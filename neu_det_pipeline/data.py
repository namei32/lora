from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Sequence

import cv2
import numpy as np
from rich.progress import track
from sklearn.model_selection import train_test_split

ANNOT_SUFFIX = ".xml"
IMG_SUFFIX = ".jpg"


@dataclass
class DefectSample:
    image_path: Path
    annotation_path: Path
    cls_name: str
    bbox: tuple[int, int, int, int]


@dataclass
class DatasetSplits:
    train: List[DefectSample]
    val: List[DefectSample]


def load_pascal_voc_annotation(xml_path: Path) -> tuple[str, tuple[int, int, int, int]]:
    import xml.etree.ElementTree as ET

    tree = ET.parse(xml_path)
    obj = tree.find("object")
    cls = obj.findtext("name")
    bbox_el = obj.find("bndbox")
    bbox = tuple(int(bbox_el.findtext(tag)) for tag in ("xmin", "ymin", "xmax", "ymax"))
    return cls, bbox


def collect_dataset(root: Path) -> List[DefectSample]:
    img_dir = root / "IMAGES"
    ann_dir = root / "ANNOTATIONS"
    samples: List[DefectSample] = []
    for xml_path in ann_dir.glob(f"*{ANNOT_SUFFIX}"):
        cls, bbox = load_pascal_voc_annotation(xml_path)
        stem = xml_path.stem
        img_path = img_dir / f"{stem}{IMG_SUFFIX}"
        if not img_path.exists():
            continue
        samples.append(DefectSample(img_path, xml_path, cls, bbox))
    return samples


def split_dataset(samples: Sequence[DefectSample], test_size: float = 0.1, seed: int = 42) -> DatasetSplits:
    train, val = train_test_split(samples, test_size=test_size, random_state=seed, stratify=[s.cls_name for s in samples])
    return DatasetSplits(list(train), list(val))


def export_metadata(samples: Sequence[DefectSample], out_path: Path) -> None:
    data = [
        {
            "image": str(sample.image_path),
            "annotation": str(sample.annotation_path),
            "class": sample.cls_name,
            "bbox": sample.bbox,
        }
        for sample in samples
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, indent=2))


def copy_assets(samples: Sequence[DefectSample], dest_root: Path) -> None:
    img_out = dest_root / "images"
    ann_out = dest_root / "annotations"
    img_out.mkdir(parents=True, exist_ok=True)
    ann_out.mkdir(parents=True, exist_ok=True)
    for sample in track(samples, description="Copying assets"):
        shutil.copy2(sample.image_path, img_out / sample.image_path.name)
        shutil.copy2(sample.annotation_path, ann_out / sample.annotation_path.name)


def compute_class_counts(samples: Sequence[DefectSample]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for sample in samples:
        counts[sample.cls_name] = counts.get(sample.cls_name, 0) + 1
    return counts


def load_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def dataset_iterator(samples: Sequence[DefectSample]) -> Iterator[tuple[np.ndarray, DefectSample]]:
    for sample in samples:
        yield load_image(sample.image_path), sample

