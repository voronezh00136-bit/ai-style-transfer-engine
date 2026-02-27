# AI Style Transformation & Generative Engine

A specialised deep-learning pipeline for high-fidelity artistic style transfer powered by **Generative Adversarial Networks (GANs)** and **Stable Diffusion** architectures.

---

## Features

| Feature | Description |
|---|---|
| **Style Fine-Tuning** | LoRA-based adaptation of model weights to isolate and reproduce specific artistic styles (e.g. Disney animation, Van Gogh brushwork) without full retraining. |
| **High-Resolution Output** | Tile-based upscaling with Real-ESRGAN, Stable Diffusion x4, or bicubic strategies — all with feathered blending to eliminate seams and preserve detail. |
| **Latent Space Exploration** | Spherical linear interpolation (slerp) between style prompts, multi-prompt journeys, and latent arithmetic for steered content generation. |

---

## Tech Stack

- **Deep Learning** — PyTorch, Hugging Face Diffusers, PEFT (LoRA)
- **Computer Vision** — OpenCV, Pillow
- **Infrastructure** — CUDA-accelerated inference; automatic CPU fallback

---

## Installation

```bash
pip install -r requirements.txt
```

For Real-ESRGAN support (optional):

```bash
pip install realesrgan basicsr
```

---

## Quick Start

### Style Transfer

```python
from PIL import Image
from style_transfer import StyleTransferEngine

engine = StyleTransferEngine()                # defaults to Stable Diffusion img2img
content = Image.open("photo.jpg")
styled  = engine.transfer(
    content,
    style_prompt="Van Gogh starry night oil painting",
    strength=0.65,
    seed=42,
)
styled.save("output.jpg")
```

Switch to the GAN backend by passing `backend="gan"` and a local checkpoint path.

---

### Style Fine-Tuning (LoRA)

```python
from PIL import Image
from style_transfer import StyleFineTuner

tuner = StyleFineTuner()
style_images = [Image.open(p) for p in ["ref1.jpg", "ref2.jpg", "ref3.jpg"]]

tuner.prepare(style_images, style_name="disney_animation").train("./adapters/disney")
```

Apply the saved adapter to any pipeline:

```python
from diffusers import StableDiffusionImg2ImgPipeline
from style_transfer import StyleFineTuner

pipe = StableDiffusionImg2ImgPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
StyleFineTuner.apply(pipe, "./adapters/disney")
```

---

### High-Resolution Upscaling

```python
from PIL import Image
from style_transfer import HighResUpscaler

upscaler = HighResUpscaler(method="bicubic")          # or "real_esrgan" / "diffusion"
low_res  = Image.open("styled_512.jpg")
high_res = upscaler.upscale(low_res, scale=4)
high_res.save("styled_2048.jpg")
```

For very large images use the tiled mode to avoid OOM errors:

```python
high_res = upscaler.tile_upscale(low_res, scale=4, tile_size=512, overlap=64)
```

---

### Latent Space Exploration

```python
from style_transfer import LatentSpaceExplorer

explorer = LatentSpaceExplorer()

# Interpolate between two styles (returns a list of PIL images)
frames = explorer.interpolate_styles(
    start_prompt="Van Gogh starry night",
    end_prompt="Disney animation style",
    steps=8,
    seed=0,
)
for i, frame in enumerate(frames):
    frame.save(f"frame_{i:02d}.png")

# Multi-prompt journey
frames = explorer.interpolate_prompts(
    prompts=["impressionist painting", "cubist artwork", "watercolour sketch"],
    steps_between=6,
)
```

---

## Project Layout

```
style_transfer/
├── __init__.py          # Public API exports
├── engine.py            # StyleTransferEngine  (GAN + Diffusion backends)
├── fine_tuning.py       # StyleFineTuner       (LoRA fine-tuning)
├── upscaler.py          # HighResUpscaler      (Real-ESRGAN / Diffusion / bicubic)
└── latent_explorer.py   # LatentSpaceExplorer  (slerp interpolation)
tests/
├── test_engine.py
├── test_fine_tuning.py
├── test_upscaler.py
└── test_latent_explorer.py
```

---

## Development

```bash
pip install -e ".[dev]"
pytest            # run all 73 tests
ruff check .      # lint
```
