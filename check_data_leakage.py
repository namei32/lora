#!/usr/bin/env python3
"""
检查混合数据集是否存在数据泄露
检查训练集中生成图片对应的原图是否在测试集或验证集中
"""

import json
from pathlib import Path


def check_data_leakage(mixed_dataset_dir, manifest_path):
    """检查数据泄露"""
    mixed_dataset_dir = Path(mixed_dataset_dir)
    manifest_path = Path(manifest_path)
    
    # 读取split_manifest.json
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    train_set = set(manifest.get("train", []))
    val_set = set(manifest.get("val", []))
    test_set = set(manifest.get("test", []))
    
    print(f"[INFO] 数据集分布:")
    print(f"  Train: {len(train_set)} 张")
    print(f"  Val:   {len(val_set)} 张")
    print(f"  Test:  {len(test_set)} 张")
    
    # 获取训练集中的所有生成图片
    train_images_dir = mixed_dataset_dir / "images" / "train"
    if not train_images_dir.exists():
        print(f"[ERROR] 训练集图片目录不存在: {train_images_dir}")
        return
    
    # 查找所有生成图片（文件名包含 _gen）
    generated_images = []
    for img_file in train_images_dir.glob("*_gen.*"):
        stem = img_file.stem  # 例如: "crazing_1_gen"
        # 提取原始图片名称（去掉 _gen 后缀）
        original_stem = stem.replace("_gen", "")
        generated_images.append((img_file.name, original_stem))
    
    print(f"\n[INFO] 训练集中找到 {len(generated_images)} 张生成图片")
    
    # 检查数据泄露
    leaks_in_val = []
    leaks_in_test = []
    
    for gen_img_name, original_stem in generated_images:
        if original_stem in val_set:
            leaks_in_val.append((gen_img_name, original_stem))
        if original_stem in test_set:
            leaks_in_test.append((gen_img_name, original_stem))
    
    # 报告结果
    print("\n" + "="*60)
    print("数据泄露检查结果:")
    print("="*60)
    
    if leaks_in_val:
        print(f"\n[WARNING] 发现 {len(leaks_in_val)} 张生成图片对应的原图在验证集中:")
        for gen_img, orig in leaks_in_val[:10]:  # 只显示前10个
            print(f"  - {gen_img} -> {orig} (在val中)")
        if len(leaks_in_val) > 10:
            print(f"  ... 还有 {len(leaks_in_val) - 10} 个")
    else:
        print("\n[OK] 未发现生成图片对应的原图在验证集中")
    
    if leaks_in_test:
        print(f"\n[WARNING] 发现 {len(leaks_in_test)} 张生成图片对应的原图在测试集中:")
        for gen_img, orig in leaks_in_test[:10]:  # 只显示前10个
            print(f"  - {gen_img} -> {orig} (在test中)")
        if len(leaks_in_test) > 10:
            print(f"  ... 还有 {len(leaks_in_test) - 10} 个")
    else:
        print("\n[OK] 未发现生成图片对应的原图在测试集中")
    
    # 统计信息
    print("\n" + "="*60)
    print("统计摘要:")
    print("="*60)
    print(f"训练集生成图片总数: {len(generated_images)}")
    print(f"数据泄露到验证集: {len(leaks_in_val)} 张")
    print(f"数据泄露到测试集: {len(leaks_in_test)} 张")
    
    if leaks_in_val or leaks_in_test:
        print("\n[ERROR] 发现数据泄露！需要修复。")
        return False
    else:
        print("\n[SUCCESS] 未发现数据泄露，数据集安全。")
        return True


if __name__ == "__main__":
    mixed_dataset_dir = Path(r"D:\VScode\lora\outputs\generated\run_20251214_154724\mixed_dataset")
    manifest_path = Path(r"D:\VScode\lora\outputs\split_manifest.json")
    
    check_data_leakage(mixed_dataset_dir, manifest_path)


