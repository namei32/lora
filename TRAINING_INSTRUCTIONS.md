# NEU-DET LoRA + ControlNet 训练指令

## 基于 Stable Diffusion 的可控生成方案

本系统实现了论文中提出的基于 Stable Diffusion 的可控高质量缺陷图像生成方案。

### YAML 配置管理

- 根目录下提供 `config.yaml`，集中定义数据集划分、LoRA、ControlNet 等默认超参数。
- 所有 CLI 命令都会自动读取该文件；如需自定义，编写新的 YAML 并通过 `--config` 选项加载。
- 示例：

  ```powershell
  python -m neu_det_pipeline.cli --config custom_config.yaml generate `
    NEU-DET `
    outputs/guidance `
    outputs/lora/lora.safetensors `
    --output-dir outputs/generated
  ```

- 若命令行未显式提供参数（如 `--test-size`），将使用配置文件中的值。

### 四个核心组件

**Component 1 - CLIP Textual Inversion:**
- 使用 CLIP 模型的文本反转特性
- 为每个原始图像类别生成对应的提示关键词
- 输出：每个缺陷类别的专用 token（如 `<neu_crazing>`）

**Component 2 - LoRA Fine-tuning:**
- 使用低秩适应（LoRA）在缺陷数据集上微调 Stable Diffusion 1.5
- 学习钢材缺陷的风格、纹理等特征
- 输出：LoRA 权重文件

**Component 3 - Feature Extraction (HED + MiDaS):**
- HED（全局嵌套边缘检测）：获取图像的近似轮廓
- MiDaS（深度图估计）：获取图像的深度图
- 输出：每张图像的 Canny/HED/Depth 引导特征

**Component 4 - ControlNet Controllable Generation:**
- 利用三种特征进行可控生成：
  - 原始输入特征（Canny edge）
  - 轮廓特征（HED）
  - 深度图特征（MiDaS）
- 生成的缺陷位置与原图紧密对齐
- 可复用现有标注，减轻生成图像的标注工作

## 当前超参数配置

### 指定参数（论文要求）
- **LoRA Weight**: 1.0 (alpha = rank = 8)
- **训练步数**: 40 steps
- **去噪强度**: 0.2
- **ControlNet-HED 权重**: 1.0
- **ControlNet-MiDaS 权重**: 1.0

### 其余参数（Stable Diffusion 默认值）
- **推理步数**: 50 steps (SD 默认)
- **Guidance Scale**: 7.5 (SD 默认)
- **ControlNet-Canny 权重**: 1.0 (默认)
- **调度器**: PNDMScheduler (SD 默认)
- **提示词模板**: "a photo of {token}" (简单默认)
- **输出尺寸**: 512x512 像素

## 完整训练流程

### 环境准备

```powershell
# 激活环境
conda activate neu-det

# 确保已安装依赖
pip install -e .

# 设置 Hugging Face Token (如需要)
$env:HF_TOKEN = "hf_your_token_here"
```

### 步骤 1: 数据集准备和划分

```powershell
python -m neu_det_pipeline.cli prepare D:\VScode\lora\NEU-DET
```

**说明**:
- 加载 NEU-DET 数据集
- 按配置文件中的 `test_size` 划分训练/验证集（默认 0.1）
- 输出元数据到 `outputs/metadata`

### 步骤 2: Component 1 - CLIP 文本反转训练

```powershell
python -m neu_det_pipeline.cli textual-inversion D:\VScode\lora\NEU-DET --output-dir outputs/textual_inversion
```

**说明**:
- **组件 1**: 使用 CLIP 模型的文本反转特性
- 为每个缺陷类别生成对应的提示关键词
- 学习专用 token（如 `<neu_crazing>`, `<neu_inclusion>` 等）
- 默认训练 800 steps
- 输出到 `outputs/textual_inversion`

### 步骤 3: Component 3 - HED/MiDaS 引导特征提取

```powershell
python -m neu_det_pipeline.cli guidance D:\VScode\lora\NEU-DET --output-dir outputs/guidance
```

**说明**:
- **组件 3**: 使用 HED 和 MiDaS 模型提取引导特征
- 提取三种特征用于 ControlNet：
  1. **Canny 边缘**: 原始输入特征 (权重 0.8)
  2. **HED 轮廓**: 全局嵌套边缘检测，获取近似轮廓 (权重 1.0)
  3. **MiDaS 深度图**: 深度图估计 (权重 1.0)
- 为每张图像生成 `*_canny.png`, `*_hed.png`, `*_depth.png`
- 这些特征使 ControlNet 能够实现可控生成

### 步骤 3.5 (可选): 提前生成提示词

如果希望提前生成所有提示词以便检查和修改：

```powershell
python -m neu_det_pipeline.cli caption D:\VScode\lora\NEU-DET --output-file outputs/captions.json
```

**说明**:
- 使用 BLIP 模型自动分析每张缺陷图像
- 为每张图像生成描述性提示词，并结合学习到的 token
- 输出到 JSON 文件
- **注意**: 步骤 5 会自动生成提示词，此步骤为可选
### 步骤 4: Component 2 - LoRA 微调

```powershell
python -m neu_det_pipeline.cli train-lora D:\VScode\lora\NEU-DET --lora-dir outputs/lora
```

**说明**:
- **组件 2**: 使用低秩适应（LoRA）微调 Stable Diffusion 1.5
- 在缺陷数据集上学习钢材缺陷的风格、纹理等特征
- **训练步数**: 40 steps
- **学习率**: 1e-4
- **Rank**: 8, **Alpha**: 8 (实现 LoRA Weight = 1.0)
- 输出权重到 `outputs/lora/lora.safetensors`

### 步骤 5: Component 4 - ControlNet 可控生成（使用自动提示词）

**基本用法（自动生成提示词）:**
```powershell
python -m neu_det_pipeline.cli generate D:\VScode\lora\NEU-DET outputs/guidance outputs/lora/lora.safetensors --output-dir outputs/generated
```

> 日志：命令会自动在 `outputs/logs/generate_YYYYMMDD_HHMMSS.log` 保存一份运行记录，可通过 `--log-file custom.log` 指定自定义路径。

**使用预先生成的提示词文件:**
```powershell
python -m neu_det_pipeline.cli generate D:\VScode\lora\NEU-DET outputs/guidance outputs/lora/lora.safetensors --output-dir outputs/generated --caption-file outputs/captions.json
```

 python -m neu_det_pipeline.cli generate `
>>   NEU-DET `
>>   outputs/guidance `
>>   outputs/lora/lora.safetensors `
>>   --output-dir outputs/generated `
>>   --caption-file outputs/captions.json `
>>   --metrics-file outputs/metrics/metrics_latest.json `
>>   

python -m neu_det_pipeline.cli generate `
  NEU-DET `
  outputs/guidance `
  outputs/lora/lora.safetensors `
  --output-dir outputs/generated `
  --caption-file outputs/captions.json `
  --metrics-file outputs/metadata/metrics_latest.json `
  --max-samples 5  # 仅生成少量样本时可选
**优先生成特定类别（如 inclusion）:**
```powershell
python -m neu_det_pipeline.cli generate D:\VScode\lora\NEU-DET outputs/guidance outputs/lora/lora.safetensors --output-dir outputs/generated --priority-class inclusion
```

**说明**:
- **组件 4**: ControlNet 可控生成
- **自动提示词**: 如果未提供 caption 文件，会自动使用 BLIP 生成每张图像的描述性提示词
- **结合 Token**: 自动生成的提示词会结合 CLIP 学习到的专用 token（如 `<neu_crazing>`）
- 利用三种特征（原始输入、轮廓、深度图）进行可控生成
- **推理步数**: 50 steps (SD 默认)
- **Guidance Scale**: 7.5 (SD 默认)
- **去噪强度**: 0.2 (指定值)
- **Control Scales**: [1.0, 1.0, 1.0] (HED=1.0, MiDaS=1.0 为指定值，Canny 采用默认)
- 生成的缺陷位置与原图紧密对齐，可复用现有标注
- 输出到 `outputs/generated`，保持 512x512 原始分辨率

## 一键运行完整流程

```powershell
# 设置数据集路径
$DATASET_ROOT = "D:\VScode\lora\NEU-DET"

# 步骤 1-5 连续执行（自动生成提示词）
python -m neu_det_pipeline.cli prepare $DATASET_ROOT
python -m neu_det_pipeline.cli textual-inversion $DATASET_ROOT --output-dir outputs/textual_inversion
python -m neu_det_pipeline.cli guidance $DATASET_ROOT --output-dir outputs/guidance
python -m neu_det_pipeline.cli train-lora $DATASET_ROOT --lora-dir outputs/lora
python -m neu_det_pipeline.cli generate $DATASET_ROOT outputs/guidance outputs/lora/lora.safetensors --output-dir outputs/generated
python prepare_yolo_dataset.py D:\VScode\lora\NEU-DET outputs/generated `
    --output-dir outputs/yolo_dataset --test-ratio 0.1 --val-ratio 0.1 `
    --generated-size 200 --seed 42
```

**说明**:
- 最后一步会自动使用 BLIP 生成每张图像的描述性提示词
- 提示词会自动结合 CLIP 学习到的 token（如 `<neu_crazing>`）
- 提示词文件会保存到 `outputs/captions.json` 供后续使用

## 超参数说明

### 指定参数（论文要求）
- **LoRA Weight**: 1.0 (通过设置 alpha = rank = 8 实现)
- **LoRA Steps**: 40 (快速微调，适合小数据集)
- **Denoising Strength**: 0.2 (轻度去噪，保持原始结构)
- **ControlNet-HED Weight**: 1.0 (完全应用轮廓特征)
- **ControlNet-MiDaS Weight**: 1.0 (完全应用深度信息)

### Stable Diffusion 默认参数
- **Inference Steps**: 50 (SD 1.5 默认推理步数)
- **Guidance Scale**: 7.5 (SD 1.5 默认文本引导强度)
- **Scheduler**: PNDMScheduler (SD 1.5 默认调度器)
- **ControlNet-Canny Weight**: 1.0 (默认权重)
- **Prompt Template**: "a photo of {token}" (简单模板)
- **Resolution**: 512x512 (SD 1.5 默认分辨率)

### LoRA 训练配置
- **Rank**: 8 (低秩矩阵维度)
- **Alpha**: 8 (缩放因子，与 rank 相等实现 weight = 1.0)
- **Learning Rate**: 1e-4 (LoRA 标准学习率)
- **Batch Size**: 2 (适应 GPU 内存)

## 输出结构

```
outputs/
├── metadata/
│   ├── train_metadata.json
│   └── val_metadata.json
├── textual_inversion/
│   ├── crazing_embedding.pt
│   ├── crazing_metrics.json
│   ├── inclusion_embedding.pt
│   ├── inclusion_metrics.json
│   ├── patches_embedding.pt
│   ├── patches_metrics.json
│   ├── pitted_surface_embedding.pt
│   ├── pitted_surface_metrics.json
│   ├── rolled-in_scale_embedding.pt
│   ├── rolled-in_scale_metrics.json
│   ├── scratches_embedding.pt
│   └── scratches_metrics.json
├── guidance/
│   ├── crazing_1_canny.png
│   ├── crazing_1_hed.png
│   ├── crazing_1_depth.png
│   └── ...
├── lora/
│   ├── lora.safetensors
│   └── lora_training_metrics.json
├── captions.json (自动生成的提示词)
└── generated/
    ├── crazing_1.png
    ├── crazing_2.png
    └── ...
```

## 性能优化建议

1. **GPU 内存不足**:
   ```python
   # 在 config.py 中调整
   batch_size: int = 1  # 减少批次大小
   ```

2. **加速训练**:
   ```python
   mixed_precision: str = "fp16"  # 使用混合精度（如果硬件支持）
   ```

3. **调整生成效果**:
   - 如需更多细节：增加 `num_inference_steps` (如 75-100)
   - 如需更强文本引导：调整 `guidance_scale` (8.0-10.0)
   - **注意**: 已采用 SD 默认值，一般无需调整



## 联系与支持

如有问题，请检查：
- PyTorch 和 CUDA 版本兼容性
- diffusers 版本 >= 0.17
- transformers 版本 >= 4.25
