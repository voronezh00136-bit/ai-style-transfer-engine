"""
Tests for style_transfer.upscaler.HighResUpscaler.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from PIL import Image

from style_transfer.upscaler import HighResUpscaler

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_image(w: int = 64, h: int = 64) -> Image.Image:
    return Image.new("RGB", (w, h), color=(80, 120, 200))


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


class TestInit:
    def test_default_method_is_bicubic(self):
        upscaler = HighResUpscaler()
        assert upscaler.method == "bicubic"

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="Unknown upscale method"):
            HighResUpscaler(method="super_secret_method")  # type: ignore[arg-type]

    def test_custom_model_id(self):
        upscaler = HighResUpscaler(method="diffusion", model_id="my/model")
        assert upscaler.model_id == "my/model"


# ---------------------------------------------------------------------------
# upscale() — bicubic
# ---------------------------------------------------------------------------


class TestBicubicUpscale:
    def test_output_size_correct(self):
        upscaler = HighResUpscaler(method="bicubic")
        img = _make_image(32, 24)
        result = upscaler.upscale(img, scale=2)
        assert result.size == (64, 48)

    def test_scale_4_produces_4x(self):
        upscaler = HighResUpscaler(method="bicubic")
        img = _make_image(16, 16)
        result = upscaler.upscale(img, scale=4)
        assert result.size == (64, 64)

    def test_scale_1_identity(self):
        upscaler = HighResUpscaler(method="bicubic")
        img = _make_image(32, 32)
        result = upscaler.upscale(img, scale=1)
        assert result.size == (32, 32)

    def test_invalid_scale_raises(self):
        upscaler = HighResUpscaler(method="bicubic")
        with pytest.raises(ValueError, match="scale must be >= 1"):
            upscaler.upscale(_make_image(), scale=0)

    def test_returns_pil_image(self):
        upscaler = HighResUpscaler(method="bicubic")
        result = upscaler.upscale(_make_image(), scale=2)
        assert isinstance(result, Image.Image)


# ---------------------------------------------------------------------------
# upscale() — real_esrgan fallback to bicubic when package absent
# ---------------------------------------------------------------------------


class TestRealESRGAN:
    def test_falls_back_to_bicubic_when_package_missing(self):
        upscaler = HighResUpscaler(method="real_esrgan")
        # Without the `realesrgan` package installed, the loader should
        # transparently fall back to bicubic.
        img = _make_image(16, 16)
        result = upscaler.upscale(img, scale=2)
        assert isinstance(result, Image.Image)
        assert result.size == (32, 32)


# ---------------------------------------------------------------------------
# upscale() — diffusion backend
# ---------------------------------------------------------------------------


class TestDiffusionUpscale:
    def test_upscale_calls_pipeline(self):
        upscaler = HighResUpscaler(method="diffusion")
        mock_pipe = MagicMock()
        mock_pipe.return_value.images = [_make_image(256, 256)]
        upscaler._model = mock_pipe

        img = _make_image(64, 64)
        result = upscaler.upscale(img, scale=4, prompt="sharp details", num_inference_steps=5)

        assert isinstance(result, Image.Image)
        mock_pipe.assert_called_once()
        call_kwargs = mock_pipe.call_args[1]
        assert call_kwargs["prompt"] == "sharp details"
        assert call_kwargs["num_inference_steps"] == 5


# ---------------------------------------------------------------------------
# tile_upscale()
# ---------------------------------------------------------------------------


class TestTileUpscale:
    def test_tiled_output_size_matches_scale(self):
        upscaler = HighResUpscaler(method="bicubic")
        img = _make_image(64, 64)
        result = upscaler.tile_upscale(img, scale=2, tile_size=32, overlap=4)
        assert result.size == (128, 128)

    def test_result_is_pil_image(self):
        upscaler = HighResUpscaler(method="bicubic")
        result = upscaler.tile_upscale(_make_image(32, 32), scale=2, tile_size=16, overlap=4)
        assert isinstance(result, Image.Image)

    def test_overlap_geq_tile_size_raises(self):
        upscaler = HighResUpscaler(method="bicubic")
        with pytest.raises(ValueError, match="tile_size must be greater than overlap"):
            upscaler.tile_upscale(_make_image(32, 32), scale=2, tile_size=8, overlap=8)


# ---------------------------------------------------------------------------
# _feather_mask()
# ---------------------------------------------------------------------------


class TestFeatherMask:
    def test_shape(self):
        mask = HighResUpscaler._feather_mask(32, 16)
        assert mask.shape == (16, 32, 1)

    def test_edges_less_than_centre(self):
        mask = HighResUpscaler._feather_mask(32, 32)
        centre = mask[16, 16, 0]
        corner = mask[0, 0, 0]
        assert centre > corner

    def test_values_in_range(self):
        mask = HighResUpscaler._feather_mask(64, 64)
        assert float(mask.min()) >= 0.0
        assert float(mask.max()) <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# load() idempotence
# ---------------------------------------------------------------------------


class TestLoad:
    def test_load_bicubic_returns_sentinel(self):
        upscaler = HighResUpscaler(method="bicubic")
        upscaler.load()
        assert upscaler._model == "bicubic"

    def test_load_is_idempotent(self):
        upscaler = HighResUpscaler(method="bicubic")
        upscaler.load()
        first = upscaler._model
        upscaler.load()
        assert upscaler._model is first
