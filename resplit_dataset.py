#!/usr/bin/env python3
"""
生成混合数据集：
- 读取指定的 split_manifest.json 与原始 images/labels
- 将原始数据复制到指定 run 文件夹下的 mixed_dataset
- 训练集包含生成图片和原始图片，复用原始图像标签
- 验证集和测试集只有原始图片
"""

import json
import random
from pathlib import Path
import argparse
import shutil
from PIL import Image


def load_manifest(manifest_path):
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def copy_image_and_label(src_img, src_labels_dir, dst_img_dir, dst_lbl_dir):
    stem = Path(src_img).stem
    lbl_file = src_labels_dir / f"{stem}.txt"
    # copy image
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_img, dst_img_dir / Path(src_img).name)
    if lbl_file.exists():
        shutil.copy2(lbl_file, dst_lbl_dir / lbl_file.name)


def create_mixed_dataset(orig_images_dir, orig_labels_dir, manifest_path, new_images_dir, run_output_dir=None, seed=42):
    """
    创建混合数据集：
    - 训练集：原始图片 + 生成图片（仅包含对应原始图片在train中的生成图片，防止数据泄露）
    - 验证集和测试集：只有原始图片
    """
    rng = random.Random(seed)
    orig_images_dir = Path(orig_images_dir)
    orig_labels_dir = Path(orig_labels_dir)
    manifest_path = Path(manifest_path)
    new_images_dir = Path(new_images_dir)
    
    # 如果未提供 run_output_dir，则默认写到 new_images_dir 的父目录下的 mixed_dataset
    # 例如：D:\VScode\lora\outputs\generated\run_20251214_154724 -> D:\VScode\lora\outputs\generated\run_20251214_154724\mixed_dataset
    if run_output_dir is None or str(run_output_dir).strip() == "":
        run_output_dir = new_images_dir / "mixed_dataset"
    else:
        run_output_dir = Path(run_output_dir)

    manifest = load_manifest(manifest_path)

    # Prepare output split dirs
    out_images_train = run_output_dir / "images" / "train"
    out_images_val = run_output_dir / "images" / "val"
    out_images_test = run_output_dir / "images" / "test"
    out_labels_train = run_output_dir / "labels" / "train"
    out_labels_val = run_output_dir / "labels" / "val"
    out_labels_test = run_output_dir / "labels" / "test"

    # 1) Copy original images and labels to all splits
    stem_to_split = {}
    train_orig_count = 0
    for split, (out_img_dir, out_lbl_dir) in {
        "train": (out_images_train, out_labels_train),
        "val": (out_images_val, out_labels_val),
        "test": (out_images_test, out_labels_test),
    }.items():
        stems = manifest.get(split, [])
        for stem in stems:
            stem_to_split[stem] = split
            # 原始图像在 IMAGES 根目录（不分子集），标签按 split 子目录存放
            img_file = orig_images_dir / f"{stem}.jpg"
            if not img_file.exists():
                img_file = orig_images_dir / f"{stem}.png"
                if not img_file.exists():
                    print(f"[WARN] Original image missing for {stem} in {split}")
                    continue
            src_labels_dir = orig_labels_dir / split
            copy_image_and_label(img_file, src_labels_dir, out_img_dir, out_lbl_dir)
            if split == "train":
                train_orig_count += 1

    # 2) Add all generated images to train set (复用原始标签)
    # 生成图片可能在 new_images_dir/images 或 new_images_dir 下
    images_root = new_images_dir / "images" if (new_images_dir / "images").exists() else new_images_dir
    new_imgs = sorted([p for p in images_root.glob("*.*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    rng.shuffle(new_imgs)
    
    print(f"[INFO] 找到 {len(new_imgs)} 张生成图片")
    print(f"[INFO] 标签目录: {orig_labels_dir}")
    if not orig_labels_dir.exists():
        print(f"[ERROR] 标签目录不存在: {orig_labels_dir}")
        print(f"[ERROR] 请检查 --orig_labels_dir 参数或创建标签文件目录")

    generated_count = 0
    skipped_no_manifest = 0
    skipped_no_label = 0
    skipped_data_leak = 0  # 防止数据泄露：跳过val/test中的生成图片
    for img in new_imgs:
        stem = img.stem
        # 复用原始标签：按 manifest 所属 split 定位标签
        split = stem_to_split.get(stem)
        if split is None:
            skipped_no_manifest += 1
            if skipped_no_manifest <= 5:  # 只显示前5个警告
                print(f"[WARN] Generated image {stem} not found in manifest, skipping")
            continue
        
        # 防止数据泄露：只添加对应原始图片在train中的生成图片
        # 如果原始图片在val或test中，则跳过该生成图片
        if split != "train":
            skipped_data_leak += 1
            if skipped_data_leak <= 5:  # 只显示前5个警告
                print(f"[INFO] Skipping generated image {stem} (original in {split}) to prevent data leakage")
            continue
        
        # 只有train split的生成图片才加入训练集
        label_dir = orig_labels_dir / split
        lbl_src = label_dir / f"{stem}.txt"
        
        if not lbl_src.exists():
            skipped_no_label += 1
            if skipped_no_label <= 5:  # 只显示前5个警告
                print(f"[WARN] Label file not found for {stem} in {split} (expected: {lbl_src}), skipping generated image")
            continue

        # 使用 _gen 后缀，保持原始格式；缩放生成图片到200x200
        gen_stem = f"{stem}_gen"
        original_ext = img.suffix  # 保持原始文件扩展名
        dst_img_path = out_images_train / f"{gen_stem}{original_ext}"
        
        # 缩放并保存生成图片到200x200，保持原始格式
        out_images_train.mkdir(parents=True, exist_ok=True)
        with Image.open(img) as im:
            im = im.convert("RGB")
            im = im.resize((200, 200), Image.BILINEAR)
            # 根据原始格式保存
            if original_ext.lower() in ['.png']:
                im.save(dst_img_path, "PNG")
            elif original_ext.lower() in ['.jpg', '.jpeg']:
                im.save(dst_img_path, "JPEG", quality=95)
            else:
                # 默认使用PNG格式
                im.save(dst_img_path, "PNG")

        # 复制对应标签到 _gen.txt，保持内容一致
        out_labels_train.mkdir(parents=True, exist_ok=True)
        shutil.copy2(lbl_src, out_labels_train / f"{gen_stem}.txt")
        generated_count += 1
    
    if skipped_no_manifest > 0:
        print(f"[WARN] 跳过了 {skipped_no_manifest} 张未在manifest中找到的生成图片")
    if skipped_data_leak > 0:
        print(f"[INFO] 为防止数据泄露，跳过了 {skipped_data_leak} 张对应原始图片在val/test中的生成图片")
    if skipped_no_label > 0:
        print(f"[WARN] 跳过了 {skipped_no_label} 张找不到标签文件的生成图片")
        print(f"[WARN] 请检查标签目录: {orig_labels_dir}")

    # 3) Save a manifest for this mixed dataset inside run_output_dir
    train_gen_count = generated_count
    mixed_manifest = {
        "seed": seed,
        "orig_images_dir": str(orig_images_dir),
        "orig_labels_dir": str(orig_labels_dir),
        "manifest_path": str(manifest_path),
        "new_images_dir": str(new_images_dir),
        "train_original_count": train_orig_count,
        "train_generated_count": train_gen_count,
        "train_total_count": train_orig_count + train_gen_count,
        "val_count": len(list(out_images_val.glob("*.*"))),
        "test_count": len(list(out_images_test.glob("*.*"))),
    }
    # ensure output directory exists before writing manifest
    run_output_dir.mkdir(parents=True, exist_ok=True)
    (run_output_dir / "mixed_manifest.json").write_text(
        json.dumps(mixed_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\n[SUCCESS] 混合数据集已生成到: {run_output_dir}")
    print(f"  train (原始): {train_orig_count}")
    print(f"  train (生成): {train_gen_count}")
    print(f"  train (总计): {train_orig_count + train_gen_count}")
    print(f"  val:   {mixed_manifest['val_count']}")
    print(f"  test:  {mixed_manifest['test_count']}")


def find_latest_run_dir(generated_dir):
    """查找最新的生成图片目录"""
    generated_path = Path(generated_dir)
    if not generated_path.exists():
        return None
    
    # 查找所有 run_* 目录
    run_dirs = [d for d in generated_path.iterdir() if d.is_dir() and d.name.startswith("run_")]
    if not run_dirs:
        return None
    
    # 按修改时间排序，返回最新的
    latest = max(run_dirs, key=lambda p: p.stat().st_mtime)
    return latest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成混合数据集（训练集：原始+生成；验证集/测试集：仅原始）")
    parser.add_argument("--orig_images_dir", default=r"D:\VScode\lora\NEU-DET\IMAGES",
                        help="原始图像根目录（未分子集）")
    parser.add_argument("--orig_labels_dir", default=r"D:\VScode\lora\outputs\yolo_baseline\labels",
                        help="原始标签根目录（按 split 子目录存放：train/val/test）")
    parser.add_argument("--manifest_path", default=r"D:\VScode\lora\outputs\split_manifest.json",
                        help="split_manifest.json 路径")
    parser.add_argument("--new_images_dir", default=None,
                        help="新生成图片所在目录（包含images子目录或直接包含图片）。如果未指定，自动查找最新的run目录")
    parser.add_argument("--generated_base_dir", default=r"D:\VScode\lora\outputs\generated",
                        help="生成图片的基础目录，用于自动查找最新run目录")
    parser.add_argument("--run_output_dir", default="",
                        help="混合数据集输出目录。留空则默认写到 new_images_dir/mixed_dataset")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--auto", action="store_true",
                        help="自动模式：自动查找最新的生成图片目录并生成混合数据集")

    args = parser.parse_args()
    
    # 自动模式：查找最新的run目录
    if args.auto or args.new_images_dir is None:
        latest_run = find_latest_run_dir(args.generated_base_dir)
        if latest_run is None:
            print(f"[ERROR] 未找到生成图片目录，请检查 {args.generated_base_dir}")
            exit(1)
        new_images_dir = latest_run
        print(f"[INFO] 自动检测到最新生成目录: {new_images_dir}")
    else:
        new_images_dir = args.new_images_dir
    
    create_mixed_dataset(
        args.orig_images_dir,
        args.orig_labels_dir,
        args.manifest_path,
        new_images_dir,
        args.run_output_dir,
        seed=args.seed,
    )
