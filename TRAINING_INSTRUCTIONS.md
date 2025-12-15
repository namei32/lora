
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

### Step 4 · 提示词生成

#### 方法 A：//简单拼接
```powershell
python -m neu_det_pipeline.cli caption D:\VScode\lora\NEU-DET --output-file outputs/captions.json
#### 方法B clip选择
```python -m neu_det_pipeline.cli caption D:\VScode\lora\NEU-DET `
  --output-file outputs/captions_clip_selected.json `
  --use-clip-selection ```

#### 方法 C：BLIP-2 动态提示词生成（推荐，更高质量）
python -m neu_det_pipeline.cli caption D:\VScode\lora\NEU-DET `
  --output-file outputs/captions_blip2.json `
  --use-blip2 `
  --blip2-model Salesforce/blip2-opt-2.7b `
  --combine-with-keywords
```


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

python -m neu_det_pipeline.cli generate `
  D:\VScode\lora\NEU-DET `
  outputs/guidance `
  outputs/lora/lora.safetensors `
  --output-dir outputs/generated `
  --caption-file outputs/captions_clip_selected.json --max-samples 50
```
  python -m neu_det_pipeline.cli generate `
  D:\VScode\lora\NEU-DET `
  outputs/guidance `
  outputs/lora/lora.safetensors `
  --output-dir outputs/generated `
  --caption-file outputs/captions_clip_selected.json
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
yolo val model=D:\VScode\runs\detect\train\weights\best.pt data=D:\VScode\ultralytics\ultralytics\cfg\datasets\data.yaml   split=test
yolo val model=D:\VScode\runs\detect\train2\weights\best.pt data=D:\VScode\ultralytics\ultralytics\cfg\datasets\neu_det.yaml   split=test

yolo val model=D:\VScode\runs\detect\yolov8原始数据集\weights\best.pt data=D:\VScode\ultralytics\ultralytics\cfg\datasets\neu_det.yaml   split=test


