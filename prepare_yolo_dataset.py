from __future__ import annotations

import argparse
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import xml.etree.ElementTree as ET

from PIL import Image
from rich.console import Console

from neu_det_pipeline.data import DefectSample, collect_dataset

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}
DEFAULT_GENERATED_SIZE = 200


@dataclass
class YoloEntry:
    image_path: Path
    dest_basename: str
    # List of (class_id, cx, cy, w, h) - normalized 0-1
    normalized_labels: List[tuple[int, float, float, float, float]]
    is_generated: bool = False


def voc_bbox_to_yolo(bbox: Sequence[int], width: int, height: int) -> tuple[float, float, float, float]:
    xmin, ymin, xmax, ymax = bbox
    w = max(xmax - xmin, 1)
    h = max(ymax - ymin, 1)
    cx = xmin + w / 2
    cy = ymin + h / 2
    return (
        cx / width,
        cy / height,
        w / width,
        h / height,
    )


def save_label(dest: Path, lines: List[str]) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(lines) + "\n")


def load_normalized_labels(xml_path: Path, class_to_id: Dict[str, int]) -> List[tuple[int, float, float, float, float]]:
    """
    Load labels from XML and normalize them based on the XML's width/height.
    This ensures labels are correct regardless of the target image resolution,
    assuming the image content is identical (just scaled).
    """
    tree = ET.parse(xml_path)
    size_el = tree.find("size")
    width = int(size_el.findtext("width"))
    height = int(size_el.findtext("height"))
    
    labels: List[tuple[int, float, float, float, float]] = []
    for obj in tree.findall("object"):
        cls = obj.findtext("name")
        if cls not in class_to_id:
            continue
        bbox_el = obj.find("bndbox")
        bbox = tuple(int(bbox_el.findtext(tag)) for tag in ("xmin", "ymin", "xmax", "ymax"))
        
        # Normalize using the original XML dimensions
        cx, cy, w, h = voc_bbox_to_yolo(bbox, width, height)
        labels.append((class_to_id[cls], cx, cy, w, h))
    return labels


def build_yolo_entries(
    samples: Sequence[DefectSample],
    class_to_id: Dict[str, int],
    name_suffix: str = "",
    is_generated: bool = False,
) -> List[YoloEntry]:
    entries: List[YoloEntry] = []
    for sample in samples:
        dest_base = f"{sample.image_path.stem}{name_suffix}" if name_suffix else sample.image_path.stem
        labels = load_normalized_labels(sample.annotation_path, class_to_id)
        if not labels:
            continue
        entries.append(
            YoloEntry(
                image_path=sample.image_path,
                dest_basename=dest_base,
                normalized_labels=labels,
                is_generated=is_generated,
            )
        )
    return entries


def build_generated_entries(
    generated_dir: Path,
    base_samples: Dict[str, DefectSample],
    class_to_id: Dict[str, int],
) -> Dict[str, List[YoloEntry]]:
    entries_by_stem: Dict[str, List[YoloEntry]] = defaultdict(list)
    missing: List[str] = []
    for image_path in generated_dir.glob("*"):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTS:
            continue
        stem = image_path.stem
        # If the generated file has a suffix like "_gen" or similar, we might need to strip it to find the original.
        # Assuming here that the stem matches the original stem exactly as per previous context.
        if stem not in base_samples:
            missing.append(image_path.name)
            continue
            
        sample = base_samples[stem]
        # Reuse labels from the original sample
        labels = load_normalized_labels(sample.annotation_path, class_to_id)
        if not labels:
            continue
            
        dest_basename = f"{stem}_gen"
        entries_by_stem[stem].append(
            YoloEntry(
                image_path=image_path,
                dest_basename=dest_basename,
                normalized_labels=labels,
                is_generated=True,
            )
        )
        
    if missing:
        Console().print(
            f"[yellow]警告: 找不到 {len(missing)} 个生成图像对应的标注，已跳过: {', '.join(missing[:5])}...[/yellow]"
        )
    return entries_by_stem


def export_split(entries: Sequence[YoloEntry], split_name: str, out_root: Path, generated_size: int) -> None:
    img_out = out_root / "images" / split_name
    label_out = out_root / "labels" / split_name
    img_out.mkdir(parents=True, exist_ok=True)
    label_out.mkdir(parents=True, exist_ok=True)

    for entry in entries:
        dest_img = img_out / f"{entry.dest_basename}{entry.image_path.suffix.lower()}"
        
        # Handle image resizing
        with Image.open(entry.image_path) as img:
            orig_w, orig_h = img.size
            final_img = img
            
            # Only resize if generated_size is set (>0) AND dimensions differ
            # This handles the 500x500 -> 200x200 case if generated_size=200
            if generated_size > 0 and (orig_w != generated_size or orig_h != generated_size):
                final_img = img.resize((generated_size, generated_size), Image.Resampling.LANCZOS)
            
            final_img.save(dest_img)

        # Write labels (already normalized, so resolution independent)
        label_lines = []
        for class_id, cx, cy, w, h in entry.normalized_labels:
            label_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        save_label(label_out / f"{entry.dest_basename}.txt", label_lines)


def save_data_yaml(out_root: Path, class_names: List[str]) -> None:
    lines = [
        f"path: {out_root.resolve()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        f"nc: {len(class_names)}",
        "names:",
    ]
    for idx, name in enumerate(class_names):
        lines.append(f"  {idx}: {name}")
    (out_root / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def compute_split_counts(total: int, test_ratio: float, val_ratio: float) -> Tuple[int, int, int]:
    if total <= 0:
        return 0, 0, 0

    test_count = int(round(total * test_ratio)) if test_ratio > 0 else 0
    val_count = int(round(total * val_ratio)) if val_ratio > 0 else 0

    test_count = min(test_count, total)
    remaining = total - test_count

    if test_ratio > 0 and test_count == 0:
        test_count = 1
        remaining -= 1

    val_count = min(val_count, remaining)
    if val_ratio > 0 and val_count == 0 and remaining > 1:
        val_count = 1
        remaining -= 1

    train_count = total - test_count - val_count
    if train_count <= 0:
        if val_count > 1:
            shift = 1 - train_count
            val_count = max(1, val_count - shift)
        elif test_count > 1:
            shift = 1 - train_count
            test_count = max(1, test_count - shift)
        train_count = total - test_count - val_count

    return train_count, val_count, test_count


def summarize_split(entries: Sequence[YoloEntry]) -> Dict[int, int]:
    counts: Counter[int] = Counter()
    for entry in entries:
        for cls, *_ in entry.normalized_labels:
            counts[cls] += 1
    return dict(counts)


def count_entries_per_class(entries: Sequence[YoloEntry]) -> Dict[int, int]:
    counts: Counter[int] = Counter()
    for entry in entries:
        if entry.normalized_labels:
            counts[entry.normalized_labels[0][0]] += 1
    return dict(counts)


def collect_generated_by_class(
    samples_subset: Iterable[DefectSample],
    class_to_id: Dict[str, int],
    generated_by_stem: Dict[str, List[YoloEntry]],
) -> Dict[int, List[YoloEntry]]:
    per_class: Dict[int, List[YoloEntry]] = defaultdict(list)
    for sample in samples_subset:
        cls_id = class_to_id[sample.cls_name]
        per_class[cls_id].extend(generated_by_stem.get(sample.image_path.stem, []))
    return per_class


def select_balanced_generated(
    per_class_entries: Dict[int, List[YoloEntry]],
    rng: random.Random,
    target_counts: Dict[int, int] | None = None,
) -> List[YoloEntry]:
    if not per_class_entries:
        return []

    available = {cls: len(entries) for cls, entries in per_class_entries.items() if entries}
    if not available:
        return []

    default_target = min(available.values()) if target_counts is None else None

    selected: List[YoloEntry] = []
    for cls, entries in per_class_entries.items():
        if not entries:
            continue
        target = target_counts.get(cls) if target_counts else default_target
        if target is None:
            target = len(entries)
        target = max(0, min(target, len(entries)))
        if target == len(entries):
            chosen = entries
        else:
            chosen = rng.sample(entries, target)
        selected.extend(chosen)
    return selected


def prepare(
    dataset_root: Path,
    generated_dir: Path,
    output_dir: Path,
    test_ratio: float,
    val_ratio: float,
    seed: int,
    generation_size: int,
) -> None:
    console = Console()
    samples = collect_dataset(dataset_root)
    if not samples:
        raise RuntimeError(f"在 {dataset_root} 下未找到任何样本。请确认包含 IMAGES 与 ANNOTATIONS。")

    class_names = sorted({sample.cls_name for sample in samples})
    class_to_id = {cls: idx for idx, cls in enumerate(class_names)}
    samples_by_class: Dict[str, List[DefectSample]] = defaultdict(list)
    for sample in samples:
        samples_by_class[sample.cls_name].append(sample)

    if not samples_by_class:
        raise RuntimeError("未能按类别构建数据集。请检查标注。")

    min_count = min(len(v) for v in samples_by_class.values())
    if min_count <= 0:
        raise RuntimeError("某些类别没有样本，无法构建平衡数据集。")

    console.print(
        f"每类可用真实样本最少 {min_count} 张，将按该数量对所有类别进行分层抽样。"
    )

    train_real: List[DefectSample] = []
    val_real: List[DefectSample] = []
    test_real: List[DefectSample] = []

    for idx, cls in enumerate(class_names):
        cls_samples = list(samples_by_class[cls])
        rng = random.Random(seed + idx)
        rng.shuffle(cls_samples)
        selected = cls_samples[:min_count]

        train_count, val_count, test_count = compute_split_counts(min_count, test_ratio, val_ratio)

        test_real.extend(selected[:test_count])
        val_real.extend(selected[test_count:test_count + val_count])
        train_real.extend(selected[test_count + val_count:])

    console.print(
        f"真实数据划分: 训练 {len(train_real)} 张, 验证 {len(val_real)} 张, 测试 {len(test_real)} 张。"
    )

    test_stems = {s.image_path.stem for s in test_real}

    sample_by_stem = {sample.image_path.stem: sample for sample in samples}
    generated_by_stem: Dict[str, List[YoloEntry]] = {}

    if generated_dir.exists():
        all_generated = build_generated_entries(generated_dir, sample_by_stem, class_to_id)
        generated_by_stem = {
            stem: entries
            for stem, entries in all_generated.items()
            if stem not in test_stems
        }
        excluded_count = sum(len(entries) for stem, entries in all_generated.items() if stem in test_stems)
        console.print(
            f"生成数据可用 {sum(len(v) for v in generated_by_stem.values())} 张 (已剔除测试集对应的 {excluded_count} 张以防泄露)。"
        )
    else:
        console.print(f"[yellow]未找到生成目录 {generated_dir}，训练/验证集中仅包含真实数据。[/yellow]")

    train_real_entries = build_yolo_entries(train_real, class_to_id)
    val_real_entries = build_yolo_entries(val_real, class_to_id)
    test_entries = build_yolo_entries(test_real, class_to_id)

    # Exp A: 训练集 = 100% 真实 + 100% 合成 (对应训练集的部分)
    # 不进行平衡采样，而是使用所有可用的生成数据以最大化数据量
    train_generated_map = collect_generated_by_class(train_real, class_to_id, generated_by_stem)
    train_generated: List[YoloEntry] = []
    for entries in train_generated_map.values():
        train_generated.extend(entries)

    # 验证集不应包含生成数据，以确保评估指标反映真实场景性能
    # 且生成数据的标签可能存在微小漂移，混入验证集会人为降低 mAP
    val_generated: List[YoloEntry] = []

    train_entries = train_real_entries + train_generated
    val_entries = val_real_entries + val_generated

    console.print(
        f"训练集真实样本: 图像 {count_entries_per_class(train_real_entries)}, 框 {summarize_split(train_real_entries)}"
    )
    if train_generated:
        console.print(
            f"训练集生成样本: 图像 {count_entries_per_class(train_generated)}, 框 {summarize_split(train_generated)}"
        )
    console.print(
        f"验证集真实样本: 图像 {count_entries_per_class(val_real_entries)}, 框 {summarize_split(val_real_entries)}"
    )
    if val_generated:
        console.print(
            f"验证集生成样本: 图像 {count_entries_per_class(val_generated)}, 框 {summarize_split(val_generated)}"
        )

    # 5. Export
    export_split(train_entries, "train", output_dir, generation_size)
    export_split(val_entries, "val", output_dir, generation_size)
    export_split(test_entries, "test", output_dir, generation_size)
    save_data_yaml(output_dir, class_names)

    def format_counts(name: str, entries: Sequence[YoloEntry]) -> str:
        counts = summarize_split(entries)
        parts = [f"{cls}:{counts.get(cls, 0)}" for cls in range(len(class_names))]
        return f"{name}({len(entries)}): {' '.join(parts)}"

    console.print(
        f"完成: 训练集 {len(train_entries)} 张, 验证集 {len(val_entries)} 张, 测试集 {len(test_entries)} 张。\n"
        f"类别统计 -> {format_counts('train', train_entries)}, {format_counts('val', val_entries)}, {format_counts('test', test_entries)}\n"
        f"数据保存在 {output_dir}。\n"
        f"注意: 生成图像已根据 XML 原始尺寸归一化标签，并调整为 {generation_size}x{generation_size} (如需原图请设为0)。"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="将真实+生成数据整理为 YOLO 数据集")
    parser.add_argument("dataset_root", type=Path, help="NEU-DET 数据集根目录 (包含 IMAGES/ANNOTATIONS)")
    parser.add_argument(
        "generated_dir",
        type=Path,
        help="生成图像目录 (默认: outputs/generated)",
        nargs="?",
        default=Path("outputs/generated"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/yolo_dataset"),
        help="YOLO 数据集输出目录",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="真实数据用于测试的比例 (默认 0.1)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="训练数据中划分为验证集的比例 (默认 0.1)",
    )
    parser.add_argument(
        "--generated-size",
        type=int,
        default=DEFAULT_GENERATED_SIZE,
        help="生成图像输出尺寸 (默认 200，0 表示不缩放)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="划分随机种子",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepare(
        args.dataset_root,
        args.generated_dir,
        args.output_dir,
        args.test_ratio,
        args.val_ratio,
        args.seed,
        args.generated_size,
    )


if __name__ == "__main__":
    main()
