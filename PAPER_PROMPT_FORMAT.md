# 论文风格提示词格式说明

## 概述

按照论文要求，提示词生成遵循以下流程：
1. **CLIP Textual Inversion** 从缺陷数据集生成关键词
2. **频率排序** 选择前 40% 的高频关键词
3. **组合格式** 将关键词与缺陷类别和 LoRA 权重结合

---

## 论文指定的关键词

### 基础关键词（前 40% 高频）
```
grayscale, greyscale, hotrolled steel strip, monochrome, no humans, 
surface defects, texture, rolled-in scale
```

### 类别特定关键词

| 缺陷类别 | 特定关键词 |
|---------|----------|
| crazing | fine cracks, crack networks, fracture lines |
| inclusion | embedded particles, inclusions, dark clusters |
| patches | oxidized patches, stain-like, diffuse defects |
| pitted_surface | pitting, corrosion pits, cavities |
| rolled-in_scale | oxide scale, scale streaks, residue bands |
| scratches | scratch grooves, abrasion, scratch marks |

---

## 完整提示词格式

### 格式定义
```
base_keywords, defect_specific_keywords, loRA:token:weight
```

### 具体例子

#### Crazing（裂纹）
```
grayscale, greyscale, hotrolled steel strip, monochrome, no humans, surface defects, 
texture, rolled-in scale, fine cracks, crack networks, fracture lines, loRA:neudet_crazing:1
```

#### Inclusion（夹杂物）
```
grayscale, greyscale, hotrolled steel strip, monochrome, no humans, surface defects, 
texture, rolled-in scale, embedded particles, inclusions, dark clusters, loRA:neudet_inclusion:1
```

#### Patches（斑点）
```
grayscale, greyscale, hotrolled steel strip, monochrome, no humans, surface defects, 
texture, rolled-in scale, oxidized patches, stain-like, diffuse defects, loRA:neudet_patches:1
```

---

## 代码实现

### 生成论文风格提示词

```python
from neu_det_pipeline.caption import KeywordExtractor

# 参数
class_name = "crazing"
token = "<neu_crazing>"
lora_weight = 1.0

# 使用论文关键词
extractor = KeywordExtractor()
prompt = extractor.build_paper_style_prompt(
    class_name=class_name,
    token=token,
    base_keywords=extractor.PAPER_KEYWORDS,
    defect_specific=extractor.DEFECT_SPECIFIC_KEYWORDS.get(class_name),
    lora_weight=lora_weight,
)

print(prompt)
# 输出示例:
# grayscale, greyscale, hotrolled steel strip, monochrome, no humans, 
# surface defects, texture, rolled-in scale, fine cracks, crack networks, 
# fracture lines, loRA:<neu_crazing>:1
```

### CLI 命令

```powershell
# 使用论文风格关键词生成提示词（默认）
python -m neu_det_pipeline.cli caption D:\VScode\lora\NEU-DET \
  --output-file outputs/captions.json \
  --use-paper-keywords True \
  --lora-weight 1.0
```

### 生成结果示例

`outputs/captions.json`:
```json
{
  "crazing_001": "grayscale, greyscale, hotrolled steel strip, monochrome, no humans, surface defects, texture, rolled-in scale, fine cracks, crack networks, fracture lines, loRA:<neu_crazing>:1",
  "crazing_002": "grayscale, greyscale, hotrolled steel strip, monochrome, no humans, surface defects, texture, rolled-in scale, fine cracks, crack networks, fracture lines, loRA:<neu_crazing>:1",
  "inclusion_001": "grayscale, greyscale, hotrolled steel strip, monochrome, no humans, surface defects, texture, rolled-in scale, embedded particles, inclusions, dark clusters, loRA:<neu_inclusion>:1",
  ...
}
```

---

## LoRA 权重说明

### 权重值含义
- **1.0**：完整应用 LoRA 权重（标准，论文用值）
- **0.5 - 0.9**：降低 LoRA 影响，增加生成多样性
- **1.1 - 1.5**：增强 LoRA 影响，更接近训练风格

### 使用方式

```powershell
# 标准权重 1.0
python -m neu_det_pipeline.cli caption ... --lora-weight 1.0

# 降低权重增加多样性
python -m neu_det_pipeline.cli caption ... --lora-weight 0.7

# 增强权重强化缺陷纹理
python -m neu_det_pipeline.cli caption ... --lora-weight 1.3
```

---

## 与生成流程的集成

### 完整工作流

```powershell
# Step 1: 生成论文风格提示词
python -m neu_det_pipeline.cli caption D:\VScode\lora\NEU-DET \
  --output-file outputs/captions.json \
  --use-paper-keywords True

# Step 2: 使用提示词进行 ControlNet 生成
python -m neu_det_pipeline.cli generate \
  D:\VScode\lora\NEU-DET \
  outputs/guidance \
  outputs/lora/lora.safetensors \
  --output-dir outputs/generated \
  --caption-file outputs/captions.json
```

---

## 与论文的对应关系

| 论文说明 | 代码实现 | 文件位置 |
|---------|---------|--------|
| CLIP textual inversion 生成关键词 | `CaptionGenerator.generate_with_token()` | `caption.py` |
| 按频率排序选择前 40% | `KeywordExtractor.PAPER_KEYWORDS` | `caption.py` L19-21 |
| 关键词 + defect category + LoRA weight | `KeywordExtractor.build_paper_style_prompt()` | `caption.py` L64-105 |
| 完整提示词格式 | 输出格式：`keyword1, keyword2, ..., loRA:token:weight` | `caption.py` L100-102 |

---

## 常见问题

### Q1: 为什么要使用论文指定的关键词？
**A**: 论文中的关键词是基于 NEU-DET 数据集的频率分析精选的，确保生成结果与论文对齐。

### Q2: 能否修改关键词列表？
**A**: 可以。修改 `neu_det_pipeline/caption.py` 中的 `KeywordExtractor.PAPER_KEYWORDS` 列表，但会影响复现性。

### Q3: LoRA 权重应该设置为多少？
**A**: 论文中使用 1.0（完整权重）。可根据需要调整：
- 降低权重（0.5-0.9）：生成多样性更高
- 增加权重（1.1-1.5）：缺陷特征更明显

### Q4: 如何验证生成的提示词？
**A**: 查看 `outputs/captions.json` 文件，每个样本对应一个论文格式的提示词。

### Q5: 提示词中的 `<neu_xxx>` token 是什么？
**A**: 这是 Textual Inversion Step 1 训练得到的特定于每个缺陷类别的 token，用于指导 LoRA 生成。

---

## 总结

✅ **推荐做法**：
- 使用论文指定的关键词（`--use-paper-keywords True`）
- 保持 LoRA 权重为 1.0
- 结合 HED + Depth ControlNet 进行生成

⚠️ **注意事项**：
- 提示词格式必须为 `keyword1, keyword2, ..., loRA:token:weight`
- 关键词之间用 `,` 分隔
- LoRA weight 格式为整数或浮点数（如 `loRA:token:1` 或 `loRA:token:1.0`）

