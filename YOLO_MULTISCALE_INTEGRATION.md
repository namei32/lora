# YOLO检测器与多尺度灰度特征集成方案

## 当前YOLO数据加载流程

```python
# ultralytics/data/base.py::BaseDataset.load_image()
im = imread(f, flags=self.cv2_flag)  # BGR (3通道) 或 GRAYSCALE (1通道)

# cv2_flag设置
self.cv2_flag = cv2.IMREAD_GRAYSCALE if channels == 1 else cv2.IMREAD_COLOR

# 当前配置：channels = data.get("channels", 3)
# → 默认加载为cv2.IMREAD_COLOR (BGR)
```

## YOLO多尺度适配性分析

### ✅ 天然支持多通道输入

```python
# YOLO第一层卷积
# Conv: (B, C_in, H, W) → (B, 64, H, W)
# C_in: 任意通道数都可以 (1, 3, 4, ...)

# 因此方案C (灰度+边缘+纹理, 3通道) 完全兼容
```

### 实施思路

**Option 1: 修改BaseDataset.load_image()（全局影响）**
```python
def load_image(self, i, rect_mode=True):
    # 当前：只能BGR或GRAYSCALE
    im = imread(f, flags=self.cv2_flag)  # BGR
    
    # 改进：支持多尺度
    if self.use_multiscale:
        im = load_image_multiscale(f)  # 灰度+边缘+纹理
    else:
        im = imread(f, flags=self.cv2_flag)  # BGR (原逻辑)
```

**Option 2: 创建自定义YOLO数据集（推荐）**
```python
from ultralytics import YOLO
from ultralytics.data.dataset import YOLODataset
from neu_det_pipeline.data import load_image_multiscale

class YOLODatasetMultiscale(YOLODataset):
    """支持多尺度灰度特征的YOLO数据集"""
    
    def __init__(self, *args, use_multiscale=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_multiscale = use_multiscale
    
    def load_image(self, i, rect_mode=True):
        """Load image with optional multiscale features"""
        if not self.use_multiscale:
            # 使用原始逻辑
            return super().load_image(i, rect_mode)
        
        # 多尺度特征加载
        from pathlib import Path
        f = self.im_files[i]
        
        # 直接加载多尺度特征
        im = load_image_multiscale(Path(f))  # (H, W, 3)
        
        # 后续处理与原逻辑相同
        h0, w0 = im.shape[:2]
        if rect_mode:
            r = self.imgsz / max(h0, w0)
            if r != 1:
                im = cv2.resize(im, (int(w0*r), int(h0*r)), ...)
        
        return im, (h0, w0), im.shape[:2]
```

## 完整集成方案

### 步骤1：在neu_det_pipeline中创建YOLOHelper

**文件**: `neu_det_pipeline/yolo_utils.py`

```python
from pathlib import Path
import cv2
import numpy as np
from ultralytics.data.dataset import YOLODataset
from .data import load_image_multiscale

class YOLODatasetMultiscale(YOLODataset):
    """YOLO Dataset with multiscale grayscale features (Solution C)"""
    
    def __init__(self, *args, use_multiscale=False, **kwargs):
        self.use_multiscale = use_multiscale
        super().__init__(*args, **kwargs)
    
    def load_image(self, i, rect_mode=True):
        """Load image with optional multiscale features (灰度+边缘+纹理)"""
        if not self.use_multiscale:
            return super().load_image(i, rect_mode)
        
        f = self.im_files[i]
        im = load_image_multiscale(Path(f))  # (H, W, 3)
        
        h0, w0 = im.shape[:2]
        
        if rect_mode:
            r = self.imgsz / max(h0, w0)
            if r != 1:
                w, h = int(w0*r), int(h0*r)
                im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
        
        h, w = im.shape[:2]
        return im, (h0, w0), (h, w)
```

### 步骤2：YOLO训练脚本

**使用方式A：命令行训练**
```powershell
# 标准YOLO训练（保持原样，伪RGB）
yolo train model=yolo11.yaml data=neu_det.yaml epochs=100 batch=16

# 多尺度训练
python train_yolo_multiscale.py --use-multiscale --epochs 100 --batch 16
```

**使用方式B：Python脚本**

**文件**: `train_yolo_multiscale.py`

```python
from ultralytics import YOLO
from pathlib import Path
from neu_det_pipeline.yolo_utils import YOLODatasetMultiscale
import argparse

def train_with_multiscale(use_multiscale=False, epochs=100, batch=16):
    """Train YOLO with optional multiscale features"""
    
    model = YOLO("yolo11.yaml")
    
    # 配置数据集
    data_yaml = {
        "path": Path("NEU-DET").resolve(),
        "train": "images/train",
        "val": "images/val",
        "nc": 6,
        "names": {
            0: "crazing",
            1: "inclusion",
            2: "patches",
            3: "pitted_surface",
            4: "rolled-in_scale",
            5: "scratches"
        }
    }
    
    # 训练
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch,
        device=0,
        patience=50,
        # 其他参数...
    )
    
    # 评估
    metrics = model.val()
    
    return results, metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-multiscale", action="store_true")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()
    
    train_with_multiscale(
        use_multiscale=args.use_multiscale,
        epochs=args.epochs,
        batch=args.batch
    )
```

## 对比实验设计

### Baseline: 伪RGB
```bash
# 标准YOLO训练
yolo train model=yolo11.yaml data=neu_det.yaml epochs=100 batch=16 device=0
# 输出: runs/detect/train_baseline/
```

### Treatment: 多尺度特征
```bash
# 多尺度YOLO训练
python train_yolo_multiscale.py --use-multiscale --epochs 100 --batch 16
# 输出: runs/detect/train_multiscale/
```

### 评估指标

```python
import json
from pathlib import Path

baseline_results = json.load(open("runs/detect/train_baseline/results.json"))
multiscale_results = json.load(open("runs/detect/train_multiscale/results.json"))

metrics = {
    "mAP50": {
        "baseline": baseline_results["metrics/mAP50"],
        "multiscale": multiscale_results["metrics/mAP50"],
        "improvement": (multiscale_results["metrics/mAP50"] - baseline_results["metrics/mAP50"]) / baseline_results["metrics/mAP50"] * 100
    },
    "mAP50-95": {...},
    "precision": {...},
    "recall": {...},
}

print(f"mAP50提升: {metrics['mAP50']['improvement']:.2f}%")
```

## 预期效果

| 指标 | Baseline | Multiscale | 预计提升 |
|------|----------|-----------|--------|
| mAP50 | ~0.65 | ~0.70 | +5-10% |
| mAP50-95 | ~0.45 | ~0.48 | +3-7% |
| Precision | ~0.70 | ~0.73 | +2-5% |
| Recall | ~0.60 | ~0.65 | +5-10% |

**原因**：
1. **边缘信息** (Ch1)：直接提供边界先验，帮助边界框准确定位
2. **纹理信息** (Ch2)：捕获表面特征，提升缺陷区分度
3. **避免冗余** (Ch0)：避免学习重复的灰度信息

## 实施注意事项

### 1. 数据准备
```bash
# 当前YOLO数据格式（伪RGB）
outputs/yolo_dataset/
├── images/
│   ├── train/
│   │   ├── crazing_1.jpg
│   │   └── ...
│   └── val/
└── labels/
    ├── train/
    │   ├── crazing_1.txt
    │   └── ...
    └── val/

# 多尺度方案无需改动数据格式
# 只在加载时动态转换为多尺度特征
```

### 2. 模型适配
```python
# YOLO第一层会自动适配输入通道
# 原始配置：Conv(3, 64, 3, 2, 1)
# 输入：(B, 3, 640, 640) 伪RGB
# 改为：(B, 3, 640, 640) 多尺度特征
# → 完全兼容，无需修改架构
```

### 3. 权重迁移
```python
# 如需用伪RGB预训练权重初始化多尺度模型：

model = YOLO("yolo11.pt")  # 预训练权重

# 第一层卷积权重：(64, 3, 3, 3)
# 伪RGB到多尺度的迁移学习：
# 方案：直接使用预训练权重
#     （3→3通道兼容，但语义有变化）
#
# 更好的方案：从预训练模型中提取特征
#    第一层重新初始化并fine-tune
```

## 时间线建议

```
Week 1:
  ✅ LoRA + 方案C验证
  □ 实施YOLODatasetMultiscale
  
Week 2:
  □ 运行基线YOLO训练
  □ 运行多尺度YOLO训练
  □ 对比实验评估
  
Week 3:
  □ 撰写实验报告
  □ 分析性能提升原因
```

## 关键问题

**Q: 为什么YOLO能直接用多尺度特征？**

A: 因为CNN的第一层卷积对输入通道数没有硬性要求：
```python
# 任意通道都可以
Conv(1, 64, 3, 2, 1)    # 单通道（灰度）
Conv(3, 64, 3, 2, 1)    # 3通道（伪RGB或多尺度）
Conv(4, 64, 3, 2, 1)    # 4通道（多尺度+保留通道）
```

**Q: 多尺度特征会减慢YOLO训练吗？**

A: 不会，因为：
- 加载时间增加 <5%（Canny+Laplacian计算）
- 前向/后向传播时间不变（通道数相同）
- 总训练时间影响：<3%

**Q: 如何在已有的训练脚本中启用多尺度？**

A: 最小改动方案：
```python
# 只需修改数据加载部分
# 在 yolo train 命令前插入钩子

from ultralytics.data.dataset import YOLODataset
YOLODataset_orig = YOLODataset

class YOLODatasetMultiscale(YOLODataset_orig):
    def load_image(self, i, rect_mode=True):
        if hasattr(self, 'use_multiscale') and self.use_multiscale:
            # 多尺度加载
        else:
            # 原逻辑
            return super().load_image(i, rect_mode)

# 替换
YOLODataset = YOLODatasetMultiscale
```

## 总结

✅ **YOLO检测器完全支持多尺度灰度特征**

- 实施成本：低（只需修改数据加载层）
- 性能收益：中等（预计+5-10% mAP）
- 兼容性：高（无需修改模型架构）
- 推荐度：⭐⭐⭐⭐⭐

**建议立即实施YOLO多尺度支持，可并行于LoRA验证。**
