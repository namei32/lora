
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
python -m neu_det_pipeline.cli generate D:\VScode\lora\NEU-DET D:\VScode\lora\outputs\guidance outputs/lora/lora.safetensors --mode a1 --use-bbox-mask


python -m neu_det_pipeline.cli generate `
  D:\VScode\lora\NEU-DET `
  D:\VScode\lora\outputs\guidance `
  D:\VScode\lora\outputs\lora\lora.safetensors `
  --mode a2 `
  --use-bbox-mask `


python -m neu_det_pipeline.cli generate `
  D:\VScode\lora\NEU-DET `
  D:\VScode\lora\outputs\guidance `
  D:\VScode\lora\outputs\lora\lora.safetensors `
  --mode a3 `
  --use-bbox-mask `



### Step 6 · 评估与下游使用
- 最新指标始终写入当前运行目录（`run_xxx/metrics.json`），包含 FID/KID/LPIPS/Edge-SSIM 及运行配置，方便逐次对比。
- 若要合并到检测训练，可结合 `prepare_yolo_dataset.py` 或自定义脚本，将 `run_xxx/images` 与对应 XML/JSON 标签打包。

yolo train model=yolov8.yaml data=D:\VScode\ultralytics\ultralytics\cfg\datasets\data.yaml epochs=1000 patience=50   batch=16  
yolo train model=rtdetr-resnet50.yaml data=D:\VScode\ultralytics\ultralytics\cfg\datasets\data.yaml epochs=1000 patience=50   batch=16  workers=0
yolo train model=rtdetr-resnet50.yaml data=D:\VScode\ultralytics\ultralytics\cfg\datasets\neu_det.yaml epochs=1000 patience=50   batch=16  workers=0
yolo val model=D:\VScode\runs\detect\b\weights\best.pt  data=D:\VScode\ultralytics\ultralytics\cfg\datasets\b.yaml   split=test
yolo val model=D:\VScode\runs\detect\train\weights\best.pt data=D:\VScode\ultralytics\ultralytics\cfg\datasets\neu_det.yaml   split=test

yolo val model=D:\VScode\runs\detect\yolov8原始数据集\weights\best.pt data=D:\VScode\ultralytics\ultralytics\cfg\datasets\neu_det.yaml   split=test


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
yolo train model=yolov8.yaml data=D:\VScode\ultralytics\ultralytics\cfg\datasets\a.yaml epochs=1000 patience=50   batch=16 

yolo train model=yolov8.yaml data=D:\VScode\ultralytics\ultralytics\cfg\datasets\b.yaml epochs=1000 patience=50   batch=16 

yolo train model=yolov8.yaml data=D:\VScode\ultralytics\ultralytics\cfg\datasets\data.yaml epochs=1000 patience=50   batch=16 