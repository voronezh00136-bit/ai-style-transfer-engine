# AI Style Transfer Engine

## Project Overview

Neural style transfer engine that transforms images using the artistic style of any reference image. Built with PyTorch as the primary deep learning framework. The project is in early development.

## Tech Stack

- **Language**: Python 3.10+
- **Deep Learning**: PyTorch (primary), TensorFlow (compatibility only)
- **Image Processing**: OpenCV, Pillow
- **Data**: NumPy, Pandas
- **Model Architectures**: VGG-19, ResNet

## Project Structure

```
models/          # Neural network architectures (VGG, ResNet, GANs)
utils/           # Image loading, preprocessing, visualization helpers
data/            # Sample images and dataset references
tests/           # pytest test suite
configs/         # YAML/JSON configuration files for training and inference
weights/         # Trained model weights (gitignored)
outputs/         # Generated images (gitignored)
transfer.py      # Main entry point
```

## Development Commands

```bash
pip install -r requirements.txt   # Install dependencies
python transfer.py --content image.jpg --style style.jpg --output result.jpg
pytest tests/                      # Run tests
ruff check .                       # Lint
mypy .                             # Type check
```

## Code Style

- PEP 8, max line length 100
- Type hints on all public functions
- Google-style docstrings
- Prefer PyTorch APIs for new model code

## Architecture Guidelines

- Model definitions go in `models/`
- Shared utilities (image I/O, transforms, metrics) go in `utils/`
- Training/inference hyperparameters go in `configs/` as YAML files
- Trained weights go in `weights/` (never committed)
- All output images go in `outputs/` (never committed)

## Remote Control Notes

When working via remote control, GPU-dependent operations (training, inference) cannot be tested. Focus remote sessions on code generation, review, refactoring, and planning.

## Frontend / Demo UI

The core engine is a Python (PyTorch) library and CLI — it has no frontend yet. If a demo UI is added (e.g. a Gradio/Streamlit playground or a React app for uploading a content image + style image and viewing the result), put it under `web/` or `demo/`.

The official `frontend-design` plugin (install via `/plugin install frontend-design@claude-plugins-official`) applies **only** to that demo UI work. It does not activate on the model/training/inference code in `models/`, `utils/`, or `transfer.py`.
