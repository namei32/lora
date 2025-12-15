# NEU-DET 训练流程技术改进方案

## 概述

本文档针对 `TRAINING_INSTRUCTIONS.md` 中 Step 0-5 的流程，提供具体的技术改进方案，包括模型升级、提示词优化、训练策略改进等。

---

## 一、模型升级方案

### 1.1 基础生成模型升级

#### 当前状态
- **SD 1.5** (`runwayml/stable-diffusion-v1-5`)
- 分辨率：512×512
- 参数量：~860M

#### 改进方案 A：升级到 SDXL（推荐）
```python
# config.py 修改
@dataclass
class GenerationConfig:
    base_model: str = "stabilityai/stable-diffusion-xl-base-1.0"  # 升级到 SDXL
    # ...
    resolution: int = 1024  # SDXL 支持更高分辨率
```

**优势**：
- 分辨率提升至 1024×1024，细节更丰富
- 参数量 2.6B，表达能力更强
- 更好的文本理解能力
- 对工业缺陷细节的捕捉更准确

**注意事项**：
- 需要升级 ControlNet 到 SDXL 版本
- 显存需求增加（建议 16GB+）
- LoRA rank 可能需要调整（建议 16-32）

#### 改进方案 B：升级到 SD 3.0 Medium（实验性）
```python
base_model: str = "stabilityai/stable-diffusion-3-medium-diffusers"
```

**优势**：
- 最新架构，性能最优
- 更强的文本-图像对齐能力
- 支持更长的提示词

**注意事项**：
- 需要 HuggingFace token
- 显存需求更高（建议 24GB+）
- ControlNet 支持可能不完整

#### 实施步骤
1. 修改 `config.py` 中的 `base_model` 参数
2. 更新 `textual_inversion.py` 和 `lora_train.py` 中的模型加载代码
3. 调整训练超参数（学习率、batch size 等）
4. 重新训练 Textual Inversion 和 LoRA

---

### 1.2 ControlNet 升级

#### 当前状态
- HED: `lllyasviel/control_v11p_sd15_softedge`
- Depth: `lllyasviel/control_v11f1p_sd15_depth`

#### 改进方案 A：升级到最新版本（SD 1.5）
```python
controlnet_models: List[str] = [
    "lllyasviel/control_v11p_sd15_softedge",  # 已是最新
    "lllyasviel/control_v11f1p_sd15_depth",   # 已是最新
    "lllyasviel/control_v11p_sd15_normalbae", # 新增：法线图（可选）
]
```

#### 改进方案 B：升级到 SDXL ControlNet（配合 SDXL 使用）
```python
controlnet_models: List[str] = [
    "diffusers/controlnet-canny-sdxl-1.0",
    "diffusers/controlnet-depth-sdxl-1.0",
]
```

#### 改进方案 C：添加更多控制模态
```python
# 添加法线图（Normal Map）控制
controlnet_models: List[str] = [
    "lllyasviel/control_v11p_sd15_softedge",   # HED
    "lllyasviel/control_v11f1p_sd15_depth",    # Depth
    "lllyasviel/control_v11p_sd15_normalbae",  # Normal（新增）
]
controlnet_modalities: List[str] = ["hed", "depth", "normal"]
```

**法线图优势**：
- 更好地捕捉表面几何细节
- 对缺陷的 3D 结构理解更准确
- 特别适合 pitted_surface、scratches 等类别

---

### 1.3 CLIP 模型升级

#### 当前状态
- `openai/clip-vit-large-patch14`（CLIP ViT-L/14）

#### 改进方案 A：升级到更高分辨率版本
```python
# caption.py 修改
model_name: str = "openai/clip-vit-large-patch14-336"  # 336px 输入
```

**优势**：
- 更高分辨率输入（336×336 vs 224×224）
- 对细节的捕捉更准确
- 提示词匹配精度提升

#### 改进方案 B：使用 OpenCLIP（更先进的架构）
```python
from open_clip import create_model_from_pretrained, get_tokenizer

# 使用 OpenCLIP ViT-H/14
model, preprocess = create_model_from_pretrained('hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K')
```

**优势**：
- 更大的模型（ViT-H vs ViT-L）
- 在更大数据集上训练（LAION-2B）
- 更好的零样本性能

#### 改进方案 C：使用多模型集成
```python
# 同时使用多个 CLIP 模型，取平均分数
models = [
    "openai/clip-vit-large-patch14",
    "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
]
```

---

## 二、提示词生成优化

### 2.1 当前方法分析

#### 问题
1. **静态关键词**：使用固定的论文关键词列表，缺乏动态性
2. **简单频率排序**：仅按频率选择，未考虑语义相关性
3. **缺乏上下文理解**：无法根据具体图像内容生成个性化提示词

### 2.2 改进方案 A：使用 BLIP-2 生成描述性提示词

#### 实现
```python
# 新增文件：neu_det_pipeline/blip2_caption.py
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch

class BLIP2CaptionGenerator:
    def __init__(self, model_name="Salesforce/blip2-opt-2.7b"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = Blip2Processor.from_pretrained(model_name)
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.float16
        ).to(self.device)
    
    def generate_caption(self, image: Image.Image) -> str:
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        generated_ids = self.model.generate(**inputs, max_length=50)
        caption = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return caption
```

**优势**：
- 根据图像内容动态生成描述
- 更准确地捕捉缺陷特征
- 生成的提示词更自然、更具体

**集成方式**：
```python
# caption.py 修改
def generate_with_blip2(
    self,
    samples: List[DefectSample],
    token_map: Dict[str, str],
    output_file: Optional[Path] = None,
) -> Dict[str, str]:
    blip2 = BLIP2CaptionGenerator()
    captions = {}
    
    for sample in track(samples, description="Generating BLIP-2 captions"):
        image = load_image(sample.image_path)
        base_caption = blip2.generate_caption(image)
        
        # 结合论文关键词和 BLIP-2 描述
        cls_name = sample.cls_name
        token = token_map.get(cls_name, cls_name.replace("_", " "))
        
        # 构建混合提示词
        prompt = f"{base_caption}, {token}, loRA:neudet1-v1:1"
        captions[sample.image_path.stem] = prompt
    
    return captions
```

### 2.3 改进方案 B：使用 LLaVA 生成结构化提示词

#### 实现
```python
# 新增文件：neu_det_pipeline/llava_caption.py
from transformers import LlavaProcessor, LlavaForConditionalGeneration

class LLaVACaptionGenerator:
    def __init__(self, model_name="llava-hf/llava-1.5-7b-hf"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = LlavaProcessor.from_pretrained(model_name)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.float16
        ).to(self.device)
    
    def generate_structured_prompt(
        self, 
        image: Image.Image, 
        defect_class: str
    ) -> str:
        prompt = f"Describe this {defect_class} defect on steel surface in detail, focusing on texture, shape, and visual characteristics."
        inputs = self.processor(prompt, image, return_tensors="pt").to(self.device)
        generate_ids = self.model.generate(**inputs, max_length=100)
        description = self.processor.batch_decode(generate_ids, skip_special_tokens=True)[0]
        return description
```

**优势**：
- 支持指令式提示词生成
- 可以要求模型关注特定方面（纹理、形状等）
- 生成的结构化描述更适合 ControlNet

### 2.4 改进方案 C：提示词优化与重排序

#### 实现
```python
# 新增文件：neu_det_pipeline/prompt_optimizer.py
class PromptOptimizer:
    """使用 CLIP 对候选提示词进行重排序和优化"""
    
    def __init__(self, clip_model_name="openai/clip-vit-large-patch14"):
        self.clip_model = CLIPModel.from_pretrained(clip_model_name)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
    
    def rank_prompts(
        self, 
        image: Image.Image, 
        candidate_prompts: List[str]
    ) -> List[Tuple[str, float]]:
        """对候选提示词按与图像的相似度排序"""
        inputs = self.clip_processor(
            text=candidate_prompts, 
            images=image, 
            return_tensors="pt",
            padding=True
        )
        with torch.no_grad():
            outputs = self.clip_model(**inputs)
            scores = outputs.logits_per_image[0].softmax(dim=0)
        
        ranked = sorted(
            zip(candidate_prompts, scores.tolist()),
            key=lambda x: x[1],
            reverse=True
        )
        return ranked
    
    def optimize_prompt(
        self,
        base_prompt: str,
        image: Image.Image,
        defect_class: str
    ) -> str:
        """生成多个变体并选择最佳"""
        variants = [
            f"{base_prompt}, high detail, macro photography",
            f"{base_prompt}, industrial inspection, surface texture",
            f"{base_prompt}, steel defect, {defect_class}",
            f"{base_prompt}, grayscale, monochrome, no humans",
        ]
        ranked = self.rank_prompts(image, variants)
        return ranked[0][0]  # 返回最佳变体
```

### 2.5 改进方案 D：使用 LLM 生成提示词（GPT-4V / Claude）

#### 实现思路
```python
# 使用 OpenAI GPT-4V 或 Anthropic Claude
import openai
from PIL import Image
import base64

class LLMCaptionGenerator:
    def __init__(self, api_key: str, model: str = "gpt-4-vision-preview"):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
    
    def generate_prompt(self, image_path: Path, defect_class: str) -> str:
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode()
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Generate a detailed, technical prompt for Stable Diffusion to generate images similar to this {defect_class} defect on steel. Focus on texture, surface characteristics, and visual details. Format: comma-separated keywords."
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
                        }
                    ]
                }
            ],
            max_tokens=150
        )
        return response.choices[0].message.content
```

**优势**：
- 最先进的视觉-语言理解能力
- 可以生成非常精确和详细的提示词
- 支持复杂的指令和上下文理解

**注意事项**：
- 需要 API 密钥和网络连接
- 可能有成本（按 token 计费）
- 适合小规模数据集或关键样本

---

## 三、LoRA 训练优化

### 3.1 当前状态分析

#### 配置
- Rank: 8
- Alpha: 8
- Learning Rate: 1e-5
- Steps: 100
- Batch Size: 1

### 3.2 改进方案 A：使用 DoRA (Weight-Decomposed Low-Rank Adaptation)

#### 原理
DoRA 将权重更新分解为幅度（magnitude）和方向（direction），比标准 LoRA 更有效。

#### 实现
```python
# 需要安装 dora: pip install peft
from peft import LoraConfig, get_peft_model, TaskType
from peft import DoRAConfig  # DoRA 配置

# lora_train.py 修改
def prepare_pipeline(self, token_embeddings_dir: Path = None):
    # ...
    # 使用 DoRA 替代标准 LoRA
    dora_config = DoRAConfig(
        r=self.cfg.rank,
        lora_alpha=self.cfg.alpha,
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        lora_dropout=self.cfg.dropout_rate,
    )
    model = get_peft_model(unet, dora_config)
    # ...
```

**优势**：
- 在相同参数量下性能更好
- 训练更稳定
- 对学习率更不敏感

### 3.3 改进方案 B：动态 Rank 调整

#### 实现
```python
# 根据训练阶段动态调整 rank
class AdaptiveLoRATrainer:
    def __init__(self, cfg: LoRAConfig):
        self.cfg = cfg
        self.initial_rank = cfg.rank
        self.final_rank = cfg.rank * 2  # 最终 rank 翻倍
    
    def get_current_rank(self, step: int, total_steps: int) -> int:
        """线性增加 rank"""
        progress = step / total_steps
        current_rank = int(
            self.initial_rank + (self.final_rank - self.initial_rank) * progress
        )
        return current_rank
```

**优势**：
- 训练初期使用小 rank，快速学习基础特征
- 训练后期使用大 rank，捕捉细节特征
- 平衡训练效率和最终性能

### 3.4 改进方案 C：使用 8-bit 优化器

#### 实现
```python
# 安装 bitsandbytes: pip install bitsandbytes
import bitsandbytes as bnb

# lora_train.py 修改
optimizer = bnb.optim.AdamW8bit(
    model.parameters(),
    lr=self.cfg.learning_rate,
    weight_decay=0.01,
)
```

**优势**：
- 显存占用减少约 50%
- 可以使用更大的 batch size
- 训练速度略有提升

### 3.5 改进方案 D：改进的学习率调度

#### 实现
```python
# 使用 OneCycleLR 或 WarmupCosineLR
from torch.optim.lr_scheduler import OneCycleLR

scheduler = OneCycleLR(
    optimizer,
    max_lr=self.cfg.learning_rate,
    total_steps=self.cfg.steps,
    pct_start=0.1,  # 10% 用于 warmup
    anneal_strategy='cos',
)
```

**优势**：
- 更稳定的训练过程
- 更好的最终性能
- 减少过拟合风险

---

## 四、数据增强与预处理

### 4.1 当前状态
- 简单的中心裁剪和缩放
- 无数据增强

### 4.2 改进方案 A：添加训练时数据增强

#### 实现
```python
# lora_train.py 修改
import torchvision.transforms as T

class LoRADataset:
    def __init__(self, ..., use_augmentation: bool = True):
        # ...
        if use_augmentation:
            self.augmentation = T.Compose([
                T.RandomHorizontalFlip(p=0.5),
                T.RandomRotation(degrees=5),
                T.ColorJitter(brightness=0.1, contrast=0.1),
                T.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            ])
        else:
            self.augmentation = None
    
    def __getitem__(self, idx: int):
        # ...
        if self.augmentation:
            img_tensor = self.augmentation(img_tensor)
        # ...
```

**注意事项**：
- 对于缺陷检测，增强要保守（避免改变缺陷位置）
- 主要使用几何变换（旋转、平移），避免颜色变换

### 4.2 改进方案 B：使用 MixUp / CutMix（实验性）

#### 实现
```python
def mixup_data(x, y, alpha=0.2):
    """MixUp 数据增强"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam
```

**注意事项**：
- 对于缺陷检测，需要谨慎使用
- 可能破坏缺陷的完整性
- 建议仅用于数据稀缺的类别

---

## 五、调度器与生成参数优化

### 5.1 当前状态
- Scheduler: `DPMSolverMultistepScheduler`
- Inference Steps: 40
- Guidance Scale: 7.0

### 5.2 改进方案 A：使用 UniPC 调度器

#### 实现
```python
# generation.py 修改
from diffusers import UniPCMultistepScheduler

SCHEDULER_REGISTRY = {
    # ...
    "UniPCMultistepScheduler": UniPCMultistepScheduler,
}

# config.py 修改
scheduler: str = "UniPCMultistepScheduler"
num_inference_steps: int = 20  # UniPC 可以用更少步数
```

**优势**：
- 更快的推理速度（20 步 vs 40 步）
- 更好的采样质量
- 更稳定的生成结果

### 5.3 改进方案 B：动态 Guidance Scale

#### 实现
```python
# 根据缺陷类别调整 guidance scale
DEFECT_GUIDANCE_SCALES = {
    "crazing": 8.0,      # 细裂纹需要更高引导
    "inclusion": 7.5,    # 内含物需要中等引导
    "patches": 7.0,      # 斑块使用默认值
    "pitted_surface": 8.5,  # 坑洞需要更高引导
    "rolled-in_scale": 7.0,
    "scratches": 8.0,    # 划痕需要更高引导
}

def get_guidance_scale(defect_class: str, base_scale: float = 7.0) -> float:
    return DEFECT_GUIDANCE_SCALES.get(defect_class, base_scale)
```

### 5.4 改进方案 C：使用 CFG Rescale（Classifier-Free Guidance Rescale）

#### 实现
```python
# generation.py 修改
from diffusers import StableDiffusionControlNetPipeline

# 在生成时使用 CFG rescale
pipe = StableDiffusionControlNetPipeline.from_pretrained(...)

# 使用 rescale_betas_zero_snr=True
image = pipe(
    prompt=prompt,
    image=control_image,
    guidance_scale=7.0,
    num_inference_steps=40,
    generator=generator,
    controlnet_conditioning_scale=control_scale,
    # 新增：CFG rescale
    guidance_rescale=0.7,  # 0.0-1.0，降低过度引导
).images[0]
```

**优势**：
- 减少过度饱和和伪影
- 更自然的生成结果
- 特别适合高 guidance scale 的情况

---

## 六、评估指标增强

### 6.1 当前指标
- FID (Fréchet Inception Distance)
- KID (Kernel Inception Distance)
- LPIPS (Learned Perceptual Image Patch Similarity)
- Edge-SSIM

### 6.2 改进方案：添加缺陷检测特定指标

#### 实现
```python
# 新增文件：neu_det_pipeline/defect_metrics.py
class DefectSpecificMetrics:
    """缺陷检测特定的评估指标"""
    
    def compute_defect_detection_accuracy(
        self,
        generated_images: List[Image.Image],
        ground_truth_images: List[Image.Image],
        yolo_model_path: Path
    ) -> Dict[str, float]:
        """使用 YOLO 模型检测生成图像中的缺陷，计算检测准确率"""
        # 加载 YOLO 模型
        from ultralytics import YOLO
        model = YOLO(yolo_model_path)
        
        # 对生成图像进行检测
        gen_detections = []
        for img in generated_images:
            results = model(img)
            gen_detections.append(results)
        
        # 计算检测准确率、召回率等
        # ...
    
    def compute_texture_similarity(
        self,
        img1: Image.Image,
        img2: Image.Image
    ) -> float:
        """使用 LBP (Local Binary Pattern) 计算纹理相似度"""
        from skimage import feature
        import numpy as np
        
        img1_gray = np.array(img1.convert('L'))
        img2_gray = np.array(img2.convert('L'))
        
        lbp1 = feature.local_binary_pattern(img1_gray, 8, 1, method='uniform')
        lbp2 = feature.local_binary_pattern(img2_gray, 8, 1, method='uniform')
        
        hist1, _ = np.histogram(lbp1.ravel(), bins=10, range=(0, 10))
        hist2, _ = np.histogram(lbp2.ravel(), bins=10, range=(0, 10))
        
        # 计算直方图相似度（卡方距离）
        similarity = 1.0 - np.sum((hist1 - hist2) ** 2 / (hist1 + hist2 + 1e-10)) / 2.0
        return similarity
```

---

## 七、实施优先级建议

### 高优先级（立即实施）
1. **CLIP 模型升级**（方案 1.3 A）：简单、效果明显
2. **提示词优化**（方案 2.2 BLIP-2）：显著提升提示词质量
3. **UniPC 调度器**（方案 5.2）：提升生成速度和质量

### 中优先级（短期实施）
1. **DoRA 训练**（方案 3.2）：提升 LoRA 性能
2. **8-bit 优化器**（方案 3.4）：减少显存占用
3. **动态 Guidance Scale**（方案 5.3）：针对不同缺陷类别优化

### 低优先级（长期实验）
1. **SDXL 升级**（方案 1.1 A）：需要大量资源重新训练
2. **LLM 提示词生成**（方案 2.5）：成本较高，适合关键样本
3. **多模态 ControlNet**（方案 1.2 C）：需要额外的预处理步骤

---

## 八、配置示例

### 8.1 改进后的 config.py 示例

```python
@dataclass
class TextualInversionConfig:
    model_id: str = "runwayml/stable-diffusion-v1-5"  # 或升级到 SDXL
    # ... 其他参数保持不变

@dataclass
class LoRAConfig:
    model_id: str = "runwayml/stable-diffusion-v1-5"
    rank: int = 16  # 如果使用 SDXL，建议增加到 16-32
    alpha: int = 16
    learning_rate: float = 1e-5
    steps: int = 200  # 增加训练步数
    batch_size: int = 2  # 如果使用 8-bit 优化器，可以增加
    use_dora: bool = True  # 新增：使用 DoRA
    use_8bit_optimizer: bool = True  # 新增：使用 8-bit 优化器

@dataclass
class GenerationConfig:
    base_model: str = "runwayml/stable-diffusion-v1-5"  # 或升级到 SDXL
    controlnet_models: List[str] = field(
        default_factory=lambda: [
            "lllyasviel/control_v11p_sd15_softedge",
            "lllyasviel/control_v11f1p_sd15_depth",
        ]
    )
    num_inference_steps: int = 20  # 如果使用 UniPC，可以减少
    guidance_scale: float = 7.0
    scheduler: str = "UniPCMultistepScheduler"  # 新增：使用 UniPC
    guidance_rescale: float = 0.7  # 新增：CFG rescale
    use_dynamic_guidance: bool = True  # 新增：根据缺陷类别调整
```

### 8.2 改进后的 caption.py 示例

```python
class CaptionGenerator:
    def __init__(
        self,
        model_name: str = "openai/clip-vit-large-patch14-336",  # 升级到 336px
        use_blip2: bool = True,  # 新增：使用 BLIP-2
    ):
        self.use_blip2 = use_blip2
        if use_blip2:
            from .blip2_caption import BLIP2CaptionGenerator
            self.blip2 = BLIP2CaptionGenerator()
        # ... 其他初始化
    
    def generate_with_token(
        self,
        samples: List[DefectSample],
        token_map: Dict[str, str],
        output_file: Optional[Path] = None,
        use_paper_keywords: bool = True,
        lora_weight: float = 1.0,
    ) -> Dict[str, str]:
        captions = {}
        
        for sample in track(samples, description="Generating captions"):
            if self.use_blip2:
                # 使用 BLIP-2 生成基础描述
                image = load_image(sample.image_path)
                blip2_caption = self.blip2.generate_caption(image)
                
                # 结合论文关键词
                cls_name = sample.cls_name
                token = token_map.get(cls_name, cls_name.replace("_", " "))
                
                base_keywords = KeywordExtractor.PAPER_KEYWORDS
                defect_keywords = KeywordExtractor.DEFECT_SPECIFIC_KEYWORDS.get(
                    cls_name, []
                )
                
                # 混合提示词
                prompt = f"{blip2_caption}, {', '.join(base_keywords)}, {', '.join(defect_keywords)}, loRA:{token}:{int(lora_weight)}"
            else:
                # 回退到原始方法
                # ...
            
            captions[sample.image_path.stem] = prompt
        
        return captions
```

---

## 九、实验建议

### 9.1 A/B 测试方案
1. **基线**：当前配置（SD 1.5 + 原始提示词）
2. **实验组 A**：SD 1.5 + BLIP-2 提示词 + UniPC
3. **实验组 B**：SD 1.5 + DoRA + 8-bit 优化器
4. **实验组 C**：SDXL + 所有改进

### 9.2 评估指标
- **生成质量**：FID, KID, LPIPS
- **缺陷检测准确率**：使用 YOLO 模型评估
- **训练效率**：训练时间、显存占用
- **生成速度**：每张图像的生成时间

---

## 十、总结

本改进方案涵盖了从模型升级到提示词优化的多个方面。建议按照优先级逐步实施，并在每个阶段进行充分的实验和评估。关键改进点：

1. **模型升级**：SDXL 或更新的模型
2. **提示词优化**：BLIP-2 / LLaVA 动态生成
3. **训练优化**：DoRA、8-bit 优化器
4. **生成优化**：UniPC 调度器、CFG rescale

这些改进将显著提升生成图像的质量和训练效率。
