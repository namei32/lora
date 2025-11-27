# NEU-DET LoRA + ControlNet Augmentation Pipeline

This repo contains a reproducible workflow that expands the [NEU-DET](https://github.com/NEU-DET) steel defect dataset using text-guided diffusion.

## Pipeline Overview
1. **Textual Inversion**: learn a CLIP embedding per defect class to craft high-fidelity prompts.
2. **LoRA fine-tuning**: adapt Stable Diffusion 1.5 to NEU-DET style/texture statistics.
3. **Structural priors**: extract HED edges and MiDaS depth maps for each original image.
4. **ControlNet-guided synthesis**: generate aligned augmentations using original, edge, and depth features so that existing annotations remain valid.

## Try It
```bash
# install
pip install -e .

# run the CLI help
neu-det-pipeline --help
```

See `neu_det_pipeline/` for implementation details and sub-commands.

neu-det-pipeline prepare D:\pycharmCode\lora_sd\NEU-DET --test-size 0.1 --out-dir outputs/metadata
neu-det-pipeline textual-inversion D:\pycharmCode\lora_sd\NEU-DET --output-dir outputs/textual_inversion
neu-det-pipeline guidance D:\pycharmCode\lora_sd\NEU-DET --output-dir outputs/guidance

neu-det-pipeline train-lora D:\pycharmCode\lora_sd\NEU-DET --lora-dir outputs/lora
neu-det-pipeline generate D:\pycharmCode\lora_sd\NEU-DET outputs/guidance outputs/lora/lora.safetensors --output-dir outputs/generated

## Generation

Set `HF_TOKEN` if the ControlNet repos require authentication:

```powershell
setx HF_TOKEN "hf_your_token_here"
```

Then run:

```powershell
conda run -n neu-det python -m neu_det_pipeline.cli generate D:\pycharmCode\lora_sd\NEU-DET guidance_dir outputs\lora\lora.safetensors --output-dir outputs\generated
```
