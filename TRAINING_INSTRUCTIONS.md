# NEU-DET 生成工作流指南

面向工业缺陷扩增的 LoRA + ControlNet 流水线，聚焦“如何可靠生成可用图像”。

---

## 1. 前置要求

- **环境**：`conda activate neu-det` 后执行 `pip install -e .`
- **配置**：根目录 `config.yaml` 提供默认超参数；任意命令都可通过 `--config custom.yaml` 覆盖。
- **数据结构**：假定 NEU-DET 数据集包含 `IMAGES/` 与 `ANNOTATIONS/`。

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

### Step 4 · CLIP 提示生成（可显式或由 Step 5 自动执行）
```powershell
python -m neu_det_pipeline.cli caption D:\VScode\lora\NEU-DET --output-file outputs/captions.json
```
- CLIP (`openai/clip-vit-large-patch14`) 对多组模板打分，自动挑选最吻合缺陷纹理的提示词，并插入对应 `<neu_cls>`。
- 结果：`outputs/captions.json`（键为原始样本名，值为最终 prompt）。

### Step 5 · ControlNet 图像生成（核心）
```powershell
# 推荐：直接给绝对路径
python -m neu_det_pipeline.cli generate `
  D:\VScode\lora\NEU-DET `
  outputs/guidance `
  outputs/lora/lora.safetensors `
  --output-dir outputs/generated `
  --caption-file outputs/captions.json `
 
```

**执行细节**
1. **样本选择**：可用 `--priority-class` 先生成稀缺类别，`--max-samples` 做小批快速验证。
2. **提示词**：若未提供 `--caption-file`，命令会自动调用 Step 4 逻辑生成 `outputs/captions.json`。
3. **控制方式**：仅启用 HED + Depth ControlNet；`config.yaml` 中的 `control_scales`、`num_inference_steps`、`guidance_scale` 可自定义。
4. **输出组织**：每次运行创建 `outputs/generated/run_YYYYMMDD_HHMMSS/`，包含
   - `images/`：生成 PNG
   - `run.log`：完整日志
   - `metrics.json`：FID/KID/LPIPS/Edge-SSIM 及运行上下文
   - `run_context.json`：LoRA/ControlNet/提示词等元数据

### Step 6 · 评估与下游使用
- 最新指标始终写入当前运行目录（`run_xxx/metrics.json`），包含 FID/KID/LPIPS/Edge-SSIM 及运行配置，方便逐次对比。
- 若要合并到检测训练，可结合 `prepare_yolo_dataset.py` 或自定义脚本，将 `run_xxx/images` 与对应 XML/JSON 标签打包。

yolo train model=yolov8.yaml data=D:\VScode\ultralytics\ultralytics\cfg\datasets\data.yaml epochs=1000 patience=50   batch=16  
yolo train model=yolov8.yaml data=D:\VScode\ultralytics\ultralytics\cfg\datasets\data.yaml epochs=1000 patience=50   batch=16  
---

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
