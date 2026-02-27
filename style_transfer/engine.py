"""
style_transfer/engine.py
------------------------
Main orchestration engine for AI-based style transfer.

Supports two backends:
    * ``"diffusion"`` — Stable Diffusion img2img pipeline (default).
    * ``"gan"``       — Lightweight GAN-based feed-forward style network.

Both backends expose the same ``transfer`` method, which accepts a PIL image
and a text style prompt and returns a styled PIL image.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

Backend = Literal["diffusion", "gan"]


class StyleTransferEngine:
    """Orchestrates high-fidelity style transfer using GAN or Stable Diffusion backends.

    Args:
        model_id: Hugging Face model ID (diffusion) or local checkpoint path (GAN).
        backend: ``"diffusion"`` (default) or ``"gan"``.
        device: PyTorch device string.  Defaults to CUDA when available, else CPU.
        dtype: Floating-point precision for inference.  Defaults to ``torch.float16``
            on CUDA and ``torch.float32`` on CPU.

    Example::

        from style_transfer import StyleTransferEngine
        from PIL import Image

        engine = StyleTransferEngine()
        content = Image.open("photo.jpg")
        styled = engine.transfer(content, style_prompt="Van Gogh starry night")
        styled.save("output.jpg")
    """

    def __init__(
        self,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        backend: Backend = "diffusion",
        device: str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        self.model_id = model_id
        self.backend: Backend = backend
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.dtype = dtype if dtype is not None else (
            torch.float16 if self.device.type == "cuda" else torch.float32
        )
        self._pipeline = None
        logger.info(
            "StyleTransferEngine initialised — backend=%s  device=%s  dtype=%s",
            backend,
            self.device,
            self.dtype,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> "StyleTransferEngine":
        """Load the underlying model into memory.

        This is called automatically on the first ``transfer`` call, but can be
        invoked explicitly to control when the model is loaded (e.g. during a
        warm-up phase).

        Returns:
            ``self`` for method chaining.
        """
        if self._pipeline is not None:
            return self
        if self.backend == "diffusion":
            self._pipeline = self._load_diffusion_pipeline()
        else:
            self._pipeline = self._load_gan_pipeline()
        return self

    def transfer(
        self,
        content_image: Image.Image,
        style_prompt: str,
        strength: float = 0.6,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 50,
        negative_prompt: str = "blurry, low quality, artifacts",
        seed: int | None = None,
    ) -> Image.Image:
        """Apply ``style_prompt`` to ``content_image`` and return the styled image.

        Args:
            content_image: Input PIL image (content to stylise).
            style_prompt: Text description of the target artistic style.
            strength: How strongly to impose the style (0.0–1.0).  Higher values
                deviate further from the content image.
            guidance_scale: Classifier-free guidance scale (diffusion backend).
            num_inference_steps: Denoising steps (diffusion backend).
            negative_prompt: Conditions to suppress in the output.
            seed: Optional random seed for reproducibility.

        Returns:
            Styled PIL image at the same resolution as ``content_image``.

        Raises:
            ValueError: If ``strength`` is outside ``[0.0, 1.0]``.
        """
        if not 0.0 <= strength <= 1.0:
            raise ValueError(f"strength must be in [0.0, 1.0], got {strength}")

        self.load()

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        original_size = content_image.size  # (W, H)
        input_image = self._preprocess(content_image)

        if self.backend == "diffusion":
            result = self._run_diffusion(
                input_image,
                style_prompt,
                strength=strength,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                negative_prompt=negative_prompt,
                generator=generator,
            )
        else:
            result = self._run_gan(input_image)

        # Restore original resolution.
        result = result.resize(original_size, Image.Resampling.LANCZOS)
        return result

    def unload(self) -> None:
        """Release the model from memory and clear the GPU cache."""
        self._pipeline = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("StyleTransferEngine: model unloaded.")

    # ------------------------------------------------------------------
    # Backend loaders
    # ------------------------------------------------------------------

    def _load_diffusion_pipeline(self):
        from diffusers import StableDiffusionImg2ImgPipeline

        logger.info("Loading Stable Diffusion img2img pipeline: %s", self.model_id)
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            self.model_id,
            torch_dtype=self.dtype,
            safety_checker=None,
        )
        pipe = pipe.to(self.device)
        if self.device.type == "cuda":
            pipe.enable_attention_slicing()
        return pipe

    def _load_gan_pipeline(self):
        logger.info("Loading GAN style-transfer network from: %s", self.model_id)
        return _GANStyleNetwork(self.model_id, device=self.device, dtype=self.dtype)

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    def _preprocess(self, image: Image.Image) -> Image.Image:
        """Resize to a multiple of 8 as required by diffusion models."""
        w, h = image.size
        w = (w // 8) * 8
        h = (h // 8) * 8
        return image.resize((w, h), Image.Resampling.LANCZOS).convert("RGB")

    def _run_diffusion(
        self,
        image: Image.Image,
        prompt: str,
        *,
        strength: float,
        guidance_scale: float,
        num_inference_steps: int,
        negative_prompt: str,
        generator,
    ) -> Image.Image:
        output = self._pipeline(
            prompt=prompt,
            image=image,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            negative_prompt=negative_prompt,
            generator=generator,
        )
        return output.images[0]

    def _run_gan(self, image: Image.Image) -> Image.Image:
        return self._pipeline.forward(image)


# ---------------------------------------------------------------------------
# GAN backend — lightweight feed-forward style network
# ---------------------------------------------------------------------------


class _GANStyleNetwork:
    """Thin wrapper around a PyTorch GAN checkpoint for style transfer.

    The checkpoint is expected to contain either:
    * A full ``nn.Module`` saved via ``torch.save(model, path)`` or
    * A ``state_dict`` along with an ``architecture`` key.

    If the checkpoint is not found, the network falls back to a simple
    identity pass-through so that the rest of the pipeline still works.
    """

    def __init__(self, checkpoint_path: str, device: torch.device, dtype: torch.dtype) -> None:
        self.device = device
        self.dtype = dtype
        self._net = self._load(checkpoint_path)

    def _load(self, path: str):
        import os

        if not os.path.exists(path):
            logger.warning(
                "GAN checkpoint not found at '%s'. Using identity transform.", path
            )
            return None
        try:
            checkpoint = torch.load(path, map_location=self.device)
            if isinstance(checkpoint, torch.nn.Module):
                return checkpoint.to(device=self.device, dtype=self.dtype).eval()
            # Assume it is a state_dict inside a dict with an 'architecture' key.
            net = checkpoint.get("architecture")
            if net is not None:
                net.load_state_dict(checkpoint["state_dict"])
                return net.to(device=self.device, dtype=self.dtype).eval()
        except Exception as exc:
            logger.error("Failed to load GAN checkpoint: %s", exc)
        return None

    def forward(self, image: Image.Image) -> Image.Image:
        if self._net is None:
            return image

        import torchvision.transforms as T
        import torchvision.transforms.functional as TF

        transform = T.Compose([T.ToTensor()])
        tensor = transform(image).unsqueeze(0).to(device=self.device, dtype=self.dtype)

        with torch.inference_mode():
            output = self._net(tensor)

        output = output.squeeze(0).clamp(0.0, 1.0).float().cpu()
        return TF.to_pil_image(output)


# ---------------------------------------------------------------------------
# Numpy-based fallback used in tests / CI where PyTorch models are absent
# ---------------------------------------------------------------------------


def _pil_to_numpy(image: Image.Image) -> np.ndarray:
    return np.array(image.convert("RGB"))


def _numpy_to_pil(array: np.ndarray) -> Image.Image:
    return Image.fromarray(array.astype(np.uint8))
