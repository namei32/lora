# NEU-DET 生成工作流指南

面向工业缺陷扩增的 LoRA + ControlNet 流水线，聚焦"如何可靠生成可用图像"。

**🆕 方案C已启用**：默认使用多尺度灰度特征（灰度+边缘+纹理）解决伪RGB冗余问题。

---

## 1. 前置要求

- **环境**：`conda activate neu-det` 后执行 `pip install -e .`
- **配置**：所有参数由 `neu_det_pipeline/config.py` 的 dataclass 定义；无需外部 YAML 配置文件
- **数据结构**：假定 NEU-DET 数据集包含 `IMAGES/` 与 `ANNOTATIONS/`
- **提示词格式**：使用论文指定的关键词 + LoRA 权重格式

---

## 1.5 方案C：多尺度灰度特征（解决伪RGB冗余）

### 背景问题
NEU-DET原始图像为伪RGB格式（3个通道完全相同），导致：
- 特征冗余：3通道包含相同信息，浪费模型容量
- 训练低效：LoRA学习到的是3倍重复特征
- 如论文[34-39]所述的特征表达能力受限

### 方案C实现
自动将伪RGB转换为信息增强的3通道特征：
- **Channel 0**: 原始灰度强度（保留基础信息）
- **Channel 1**: Canny边缘图（捕获缺陷边界）
- **Channel 2**: Laplacian纹理图（捕获表面粗糙度）

### 验证方案C
运行验证脚本查看特征独立性：
```powershell
python test_multiscale_features.py
```
输出示例：
```
Channel 0 (灰度): mean=160.64, std=28.50
Channel 1 (边缘): mean=48.70, std=100.23
Channel 2 (纹理): mean=22.94, std=17.87
✓ 确认3通道独立（无冗余）
```

### 配置开关
在 `config.yaml` 中控制：
```yaml
lora:
  use_multiscale_features: true  # 启用方案C（推荐）
  # use_multiscale_features: false  # 回退到伪RGB（不推荐）
```

### 与Step 2的关系
- **不冲突**：方案C用于LoRA训练输入，Step 2用于ControlNet引导
- **互补**：方案C增强特征表达，Step 2保证生成结构一致性
- **独立**：两者作用于不同阶段，可同时使用

---

## 2. 模块速览

| 模块 | 目标 | 关键命令 |
| --- | --- | --- |
| 文本反演 | 为每个类别学习 token (`<neu_cls>`) | `textual-inversion` |
| 引导提取 | 生成 HED/Depth 控制图 | `guidance` |
| LoRA 训练 | 微调 SD 1.5 捕捉缺陷纹理 | `train-lora` |
| CLIP 提示 | 选择与图像最匹配的模板 | `caption` 或由 `generate` 自动触发 |
| ControlNet 生成 | 使用 LoRA+ControlNet 输出新图 | `generate` |
| 指标评估 | FID/KID/LPIPS/Edge-SSIM | 自动写入 `run_xxx/metrics.json` |

---

## 3. 详细生成流程

> 默认假设 `DATASET_ROOT=D:\VScode\lora\NEU-DET`，所有路径皆可自定义。

### Step 0 · 数据划分（一次即可）
```powershell
python -m neu_det_pipeline.cli prepare %DATASET_ROOT%
```
- 读取 XML + JPG，按 `config.yaml` 的 `test_size` 进行分层划分。
- 结果：`outputs/metadata/train_metadata.json` 与 `val_metadata.json`。

### Step 1 · 文本反演 Token
```powershell
python -m neu_det_pipeline.cli textual-inversion D:\VScode\lora\NEU-DET --output-dir outputs/textual_inversion
```
- 为每个缺陷类别训练 `<neu_xxx>` token，默认 800 步。
- 输出：`outputs/textual_inversion/*.pt`（供 LoRA/生成阶段使用）。

### Step 2 · 控制引导 (HED + Depth)
```powershell
python -m neu_det_pipeline.cli guidance D:\VScode\lora\NEU-DET --output-dir outputs/guidance
```
- 提取 `*_hed.png`、`*_depth.png`，用于 ControlNet。
- `*_canny.png` 仍会生成，但默认不输入生成管线。

### Step 3 · LoRA 训练
```powershell
python -m neu_det_pipeline.cli train-lora D:\VScode\lora\NEU-DET --lora-dir outputs/lora
```
- 关键超参（可在 `config.yaml` 覆盖）：rank/alpha=8、lr=1e-4、steps=40、batch_size=1~2。
- 输出：`outputs/lora/lora.safetensors` 与 `lora_training_metrics.json`、`lora_config.json`。

### Step 4 · CLIP 提示生成（论文风格关键词）
```powershell
python -m neu_det_pipeline.cli caption D:\VScode\lora\NEU-DET --output-file outputs/captions.json
```
- **论文方法**：使用 CLIP textual inversion 从缺陷数据集生成关键词
- **关键词选择**：按频率排序，选择前 40% 的高频关键词
- **提示词格式**：将关键词与缺陷类别和 LoRA 权重组合
  ```
  grayscale, greyscale, hotrolled steel strip, monochrome, no humans, 
  surface defects, texture, rolled-in scale, loRA:neudet1-v1:1
  ```
- **输出**：`outputs/captions.json`（键为样本名，值为论文风格提示词）

**论文关键词列表**（已集成）：
```
基础关键词：grayscale, greyscale, hotrolled steel strip, monochrome, no humans, surface defects, texture, rolled-in scale
类别特定：根据缺陷类型（crazing/inclusion/patches/pitted_surface/rolled-in_scale/scratches）附加相关词汇
LoRA权重：添加 "loRA:neudet1-v1:1" 格式的权重指示
```

### Step 5 · ControlNet 图像生成（核心）
```powershell
# 推荐：使用论文风格的提示词
python -m neu_det_pipeline.cli generate `
  D:\VScode\lora\NEU-DET `
  outputs/guidance `
  outputs/lora/lora.safetensors `
  --output-dir outputs/generated `
  --caption-file outputs/captions.json `
  --max-samples 50
```

**执行细节**
1. **提示词格式**：使用 Step 4 生成的论文风格提示词（包含高频关键词 + LoRA 权重）
2. **样本选择**：可用 `--priority-class` 先生成稀缺类别，`--max-samples` 做小批快速验证
3. **控制方式**：使用 HED + Depth ControlNet；当前配置：
   - `num_inference_steps`: 60（推理步数）
   - `guidance_scale`: 7.0（CFG 尺度）
   - `control_scales`: [0.7, 0.7]（HED/Midas 权重）
   - `denoising_strength`: 0.3（去噪强度）
4. **输出组织**：每次运行创建 `outputs/generated/run_YYYYMMDD_HHMMSS/`，包含
   - `images/`：生成 PNG
   - `run.log`：完整日志
   - `metrics.json`：FID/KID/LPIPS/Edge-SSIM 及运行上下文
   - `run_context.json`：LoRA/ControlNet/提示词等元数据

### Step 6 · 评估与下游使用
- 最新指标始终写入当前运行目录（`run_xxx/metrics.json`），包含 FID/KID/LPIPS/Edge-SSIM 及运行配置，方便逐次对比。
- 若要合并到检测训练，可结合 `prepare_yolo_dataset.py` 或自定义脚本，将 `run_xxx/images` 与对应 XML/JSON 标签打包。

yolo train model=yolov8.yaml data=D:\VScode\ultralytics\ultralytics\cfg\datasets\data.yaml epochs=1000 patience=50   batch=16  
yolo train model=rtdetr-resnet50.yaml data=D:\VScode\ultralytics\ultralytics\cfg\datasets\data.yaml epochs=1000 patience=50   batch=16  workers=0
yolo train model=rtdetr-resnet50.yaml data=D:\VScode\ultralytics\ultralytics\cfg\datasets\neu_det.yaml epochs=1000 patience=50   batch=16  workers=0
yolo val model=D:\VScode\runs\detect\train\weights\best.pt  data=D:\VScode\ultralytics\ultralytics\cfg\datasets\data.yaml   split=test
yolo val model=D:\VScode\runs\detect\train2\weights\best.pt data=D:\VScode\ultralytics\ultralytics\cfg\datasets\neu_det.yaml   split=test

yolo val model=D:\VScode\runs\detect\yolov8原始数据集\weights\best.pt data=D:\VScode\ultralytics\ultralytics\cfg\datasets\neu_det.yaml   split=test


# 方法D 伪彩
python -m neu_det_pipeline.cli train-lora NEU-DET --lora-dir outputs/pseudo_mode/lora

python -m neu_det_pipeline.cli generate NEU-DET outputs/guidance outputs/pseudo_mode/lora/lora.safetensors --output-dir outputs/pseudo_mode/generated 
---
# 方法B 复用三通道
# Step 2: 执行训练流程
python -m neu_det_pipeline.cli textual-inversion NEU-DET --output-dir outputs/baseline_copy3/textual_inversion

python -m neu_det_pipeline.cli guidance NEU-DET --output-dir outputs/baseline_copy3/guidance

python -m neu_det_pipeline.cli train-lora NEU-DET --lora-dir outputs/baseline_copy3/lora

python -m neu_det_pipeline.cli generate NEU-DET outputs/baseline_copy3/guidance outputs/baseline_copy3/lora/lora.safetensors --output-dir outputs/baseline_copy3/generated --max-samples 20
# 方法C 单通道
## 4. 常见参数调优

| 目标 | 建议参数 |
| --- | --- |
| 提升纹理细节 | 提高 `generation.num_inference_steps` 至 75/100；或增大 `guidance_scale` 到 8.5~10 |
| 控制噪声/过拟合 | 调低 `LoRA` 学习率；或在 `generation.control_scales` 中降低 depth 权重 |
| 快速冒烟测试 | 使用 `--max-samples 5`，并指向较小 `--output-dir` |
| 指定类别扩增 | 结合 `--priority-class` 与自定义 `caption_file` 针对性生成 |

---

## 5. 输出目录速览

```
outputs/
├── guidance/                 # HED/Depth 引导图
├── lora/                     # LoRA 权重、训练指标、配置
├── captions.json            # 最新 CLIP 提示（可被覆盖）
├── generated/
│   └── run_20251205_101500/
│       ├── images/*.png
│       ├── run.log
│       ├── metrics.json
│       └── run_context.json  # LoRA/ControlNet/提示词/路径等元信息
└── metrics/metrics_latest.json
```

---

## 6. 故障排查

1. **提示词看起来偏题**：删除 `outputs/captions.json` 让 `generate` 重新调用 CLIP；或手动编辑 JSON。
2. **显存不足**：在 `config.yaml` 中降低 `generation.num_inference_steps`、`LoRA.batch_size`，或在命令中加 `--max-samples` 分批运行。
3. **指标不升反降**：启用“Teacher Model 伪标签校验”脚本，对生成图自动清洗（详见 `validator.py` 设计）。

---

保持以上流程，可以快速对 NEU-DET 进行高质量、可追溯的缺陷图像生成。*** End Patch
