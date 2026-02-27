"""
style_transfer/upscaler.py
--------------------------
High-resolution output module.

``HighResUpscaler`` provides several strategies for producing crisp, large-scale
output images without the visual artifacts that simple bilinear/bicubic
interpolation introduces:

* ``"real_esrgan"`` — Real-ESRGAN super-resolution model (requires the
  ``realesrgan`` package to be installed).
* ``"diffusion"``   — Stable Diffusion upscaler (``stabilityai/stable-diffusion-x4-upscaler``).
* ``"bicubic"``     — High-quality PIL fallback that needs no extra dependencies.

All strategies are exposed through the same ``upscale(image, scale)`` method.
"""

from __future__ import annotations

import logging
from typing import Literal

import torch
from PIL import Image

logger = logging.getLogger(__name__)

UpscaleMethod = Literal["real_esrgan", "diffusion", "bicubic"]

_DIFFUSION_UPSCALER_ID = "stabilityai/stable-diffusion-x4-upscaler"


class HighResUpscaler:
    """Upscales images to high resolution with minimal artifact degradation.

    Args:
        method: Upscaling strategy — ``"real_esrgan"``, ``"diffusion"``, or
            ``"bicubic"`` (default).
        device: PyTorch device string.  Defaults to CUDA when available.
        dtype: Floating-point precision.  Defaults to ``float16`` on CUDA.
        model_id: Override the default Hugging Face model ID when ``method``
            is ``"diffusion"``.

    Example::

        from style_transfer import HighResUpscaler
        from PIL import Image

        upscaler = HighResUpscaler(method="bicubic")
        low_res = Image.open("styled_512.jpg")
        high_res = upscaler.upscale(low_res, scale=4)
        high_res.save("styled_2048.jpg")
    """

    def __init__(
        self,
        method: UpscaleMethod = "bicubic",
        device: str | None = None,
        dtype: torch.dtype | None = None,
        model_id: str | None = None,
    ) -> None:
        if method not in ("real_esrgan", "diffusion", "bicubic"):
            raise ValueError(
                f"Unknown upscale method '{method}'. "
                "Choose from 'real_esrgan', 'diffusion', or 'bicubic'."
            )
        self.method: UpscaleMethod = method
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.dtype = dtype or (
            torch.float16 if self.device.type == "cuda" else torch.float32
        )
        self.model_id = model_id or _DIFFUSION_UPSCALER_ID
        self._model = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> "HighResUpscaler":
        """Load the upscaling model into memory.

        Called automatically on the first ``upscale`` call.

        Returns:
            ``self`` for method chaining.
        """
        if self._model is not None:
            return self
        if self.method == "real_esrgan":
            self._model = self._load_real_esrgan()
        elif self.method == "diffusion":
            self._model = self._load_diffusion_upscaler()
        else:
            self._model = "bicubic"  # sentinel — no external model needed.
        return self

    def upscale(
        self,
        image: Image.Image,
        scale: int = 4,
        prompt: str = "high resolution, detailed, sharp",
        num_inference_steps: int = 20,
    ) -> Image.Image:
        """Return an upscaled version of ``image``.

        Args:
            image: Input PIL image to upscale.
            scale: Integer upscaling factor (e.g. 4 → 4× resolution).
            prompt: Text prompt used to guide detail generation when
                ``method="diffusion"`` (ignored otherwise).
            num_inference_steps: Denoising steps for the diffusion upscaler.

        Returns:
            Upscaled PIL image.

        Raises:
            ValueError: If ``scale`` is less than 1.
        """
        if scale < 1:
            raise ValueError(f"scale must be >= 1, got {scale}.")

        self.load()

        if self.method == "real_esrgan":
            return self._upscale_real_esrgan(image, scale)
        if self.method == "diffusion":
            return self._upscale_diffusion(image, prompt, num_inference_steps)
        return self._upscale_bicubic(image, scale)

    def tile_upscale(
        self,
        image: Image.Image,
        scale: int = 4,
        tile_size: int = 512,
        overlap: int = 64,
    ) -> Image.Image:
        """Upscale ``image`` in tiles to handle arbitrarily large inputs.

        Splits the input into overlapping tiles, upscales each tile
        independently, and blends them back together with linear feathering
        to avoid visible seams.

        Args:
            image: Input PIL image.
            scale: Integer upscaling factor.
            tile_size: Width/height of each input tile (pixels).
            overlap: Overlap between adjacent tiles (pixels) used for blending.

        Returns:
            Upscaled PIL image.
        """
        import numpy as np

        img_np = np.array(image.convert("RGB")).astype(np.float32)
        h, w = img_np.shape[:2]
        out_h, out_w = h * scale, w * scale
        canvas = np.zeros((out_h, out_w, 3), dtype=np.float32)
        weight = np.zeros((out_h, out_w, 1), dtype=np.float32)

        step = tile_size - overlap
        if step <= 0:
            raise ValueError("tile_size must be greater than overlap.")

        for y in range(0, h, step):
            for x in range(0, w, step):
                x1 = min(x, w - tile_size) if w > tile_size else 0
                y1 = min(y, h - tile_size) if h > tile_size else 0
                x2 = min(x1 + tile_size, w)
                y2 = min(y1 + tile_size, h)

                tile = image.crop((x1, y1, x2, y2))
                tile_up = self.upscale(tile, scale=scale)
                tile_np = np.array(tile_up).astype(np.float32)

                ox1, oy1 = x1 * scale, y1 * scale
                ox2, oy2 = ox1 + tile_np.shape[1], oy1 + tile_np.shape[0]

                # Feathering mask.
                mask = self._feather_mask(tile_np.shape[1], tile_np.shape[0])
                canvas[oy1:oy2, ox1:ox2] += tile_np * mask
                weight[oy1:oy2, ox1:ox2] += mask

        weight = np.maximum(weight, 1e-6)
        result_np = np.clip(canvas / weight, 0, 255).astype(np.uint8)
        return Image.fromarray(result_np)

    # ------------------------------------------------------------------
    # Backend implementations
    # ------------------------------------------------------------------

    def _load_real_esrgan(self):
        try:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer

            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32)
            upsampler = RealESRGANer(
                scale=4,
                model_path="https://github.com/xinntao/Real-ESRGAN/releases/download/"
                "v0.1.0/RealESRGAN_x4plus.pth",
                model=model,
                device=self.device,
            )
            logger.info("Real-ESRGAN loaded.")
            return upsampler
        except ImportError:
            logger.warning(
                "realesrgan package not found — falling back to bicubic upscaling. "
                "Install with: pip install realesrgan basicsr"
            )
            self.method = "bicubic"
            return "bicubic"

    def _load_diffusion_upscaler(self):
        from diffusers import StableDiffusionUpscalePipeline

        logger.info("Loading Stable Diffusion upscaler: %s", self.model_id)
        pipe = StableDiffusionUpscalePipeline.from_pretrained(
            self.model_id, torch_dtype=self.dtype
        ).to(self.device)
        return pipe

    def _upscale_bicubic(self, image: Image.Image, scale: int) -> Image.Image:
        w, h = image.size
        return image.resize((w * scale, h * scale), Image.Resampling.BICUBIC)

    def _upscale_real_esrgan(self, image: Image.Image, scale: int) -> Image.Image:
        import numpy as np

        if self._model == "bicubic":
            return self._upscale_bicubic(image, scale)

        img_np = np.array(image.convert("RGB"))
        out_np, _ = self._model.enhance(img_np, outscale=scale)
        return Image.fromarray(out_np)

    def _upscale_diffusion(
        self, image: Image.Image, prompt: str, num_inference_steps: int
    ) -> Image.Image:
        output = self._model(
            prompt=prompt,
            image=image,
            num_inference_steps=num_inference_steps,
        )
        return output.images[0]

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _feather_mask(width: int, height: int):
        """Create a 2-D feathering weight mask for smooth tile blending."""
        import numpy as np

        x = np.linspace(0, 1, width)
        y = np.linspace(0, 1, height)
        # Ramp up then ramp down so edges are down-weighted.
        x_mask = np.minimum(x, 1 - x) * 2
        y_mask = np.minimum(y, 1 - y) * 2
        mask_2d = np.outer(y_mask, x_mask)
        return mask_2d[:, :, np.newaxis].astype(np.float32)
