"""
Tests for style_transfer.engine.StyleTransferEngine.

All heavy model I/O is mocked so tests run on CPU without downloading weights.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch
from PIL import Image

from style_transfer.engine import StyleTransferEngine, _numpy_to_pil, _pil_to_numpy

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_image(width: int = 64, height: int = 64) -> Image.Image:
    return Image.new("RGB", (width, height), color=(128, 64, 32))


def _make_pipeline_mock(output_image: Image.Image):
    """Return a callable mock that mimics a diffusers img2img pipeline."""
    mock = MagicMock()
    result = MagicMock()
    result.images = [output_image]
    mock.return_value = result
    return mock


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


class TestStyleTransferEngineInit:
    def test_defaults(self):
        engine = StyleTransferEngine()
        assert engine.backend == "diffusion"
        assert engine.model_id == "runwayml/stable-diffusion-v1-5"
        assert engine._pipeline is None

    def test_custom_backend(self):
        engine = StyleTransferEngine(backend="gan")
        assert engine.backend == "gan"

    def test_custom_device(self):
        engine = StyleTransferEngine(device="cpu")
        assert engine.device == torch.device("cpu")

    def test_custom_dtype(self):
        engine = StyleTransferEngine(dtype=torch.float32)
        assert engine.dtype == torch.float32


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


class TestPreprocess:
    def test_rounds_down_to_multiple_of_8(self):
        engine = StyleTransferEngine()
        img = _make_image(width=65, height=73)
        result = engine._preprocess(img)
        w, h = result.size
        assert w % 8 == 0
        assert h % 8 == 0
        assert w <= 65
        assert h <= 73

    def test_converts_to_rgb(self):
        engine = StyleTransferEngine()
        rgba = Image.new("RGBA", (64, 64))
        result = engine._preprocess(rgba)
        assert result.mode == "RGB"

    def test_already_aligned_image_unchanged_dimensions(self):
        engine = StyleTransferEngine()
        img = _make_image(width=512, height=512)
        result = engine._preprocess(img)
        assert result.size == (512, 512)


# ---------------------------------------------------------------------------
# transfer()  — diffusion backend
# ---------------------------------------------------------------------------


class TestTransferDiffusion:
    def _engine_with_mock_pipeline(self) -> tuple[StyleTransferEngine, Image.Image]:
        engine = StyleTransferEngine(backend="diffusion", device="cpu", dtype=torch.float32)
        output_img = _make_image(64, 64)
        engine._pipeline = _make_pipeline_mock(output_img)
        return engine, output_img

    def test_transfer_returns_pil_image(self):
        engine, _ = self._engine_with_mock_pipeline()
        content = _make_image(64, 64)
        result = engine.transfer(content, style_prompt="Van Gogh", num_inference_steps=1)
        assert isinstance(result, Image.Image)

    def test_output_matches_input_size(self):
        engine, _ = self._engine_with_mock_pipeline()
        content = _make_image(128, 96)
        result = engine.transfer(content, style_prompt="watercolour", num_inference_steps=1)
        assert result.size == (128, 96)

    def test_invalid_strength_raises(self):
        engine, _ = self._engine_with_mock_pipeline()
        content = _make_image()
        with pytest.raises(ValueError, match="strength must be in"):
            engine.transfer(content, style_prompt="test", strength=1.5)

    def test_strength_zero_allowed(self):
        engine, _ = self._engine_with_mock_pipeline()
        content = _make_image()
        result = engine.transfer(content, style_prompt="test", strength=0.0)
        assert isinstance(result, Image.Image)

    def test_pipeline_called_with_correct_args(self):
        engine, _ = self._engine_with_mock_pipeline()
        content = _make_image(64, 64)
        engine.transfer(
            content,
            style_prompt="Monet",
            strength=0.7,
            guidance_scale=8.0,
            num_inference_steps=2,
            negative_prompt="blurry",
        )
        call_kwargs = engine._pipeline.call_args[1]
        assert call_kwargs["prompt"] == "Monet"
        assert call_kwargs["strength"] == pytest.approx(0.7)
        assert call_kwargs["guidance_scale"] == pytest.approx(8.0)
        assert call_kwargs["num_inference_steps"] == 2
        assert call_kwargs["negative_prompt"] == "blurry"

    def test_seed_produces_generator(self):
        engine, _ = self._engine_with_mock_pipeline()
        content = _make_image()
        engine.transfer(content, style_prompt="test", seed=123, num_inference_steps=1)
        call_kwargs = engine._pipeline.call_args[1]
        assert call_kwargs["generator"] is not None

    def test_no_seed_passes_none_generator(self):
        engine, _ = self._engine_with_mock_pipeline()
        content = _make_image()
        engine.transfer(content, style_prompt="test", num_inference_steps=1)
        call_kwargs = engine._pipeline.call_args[1]
        assert call_kwargs["generator"] is None


# ---------------------------------------------------------------------------
# transfer()  — GAN backend
# ---------------------------------------------------------------------------


class TestTransferGAN:
    def _engine_with_mock_gan(self) -> StyleTransferEngine:
        engine = StyleTransferEngine(backend="gan", device="cpu", dtype=torch.float32)
        mock_gan = MagicMock()
        mock_gan.forward.return_value = _make_image(64, 64)
        engine._pipeline = mock_gan
        return engine

    def test_gan_transfer_returns_pil_image(self):
        engine = self._engine_with_mock_gan()
        content = _make_image(64, 64)
        result = engine.transfer(content, style_prompt="ignored", num_inference_steps=1)
        assert isinstance(result, Image.Image)

    def test_gan_forward_called(self):
        engine = self._engine_with_mock_gan()
        content = _make_image(64, 64)
        engine.transfer(content, style_prompt="test", num_inference_steps=1)
        engine._pipeline.forward.assert_called_once()


# ---------------------------------------------------------------------------
# load() / unload()
# ---------------------------------------------------------------------------


class TestLoadUnload:
    def test_load_is_idempotent(self):
        engine = StyleTransferEngine(backend="diffusion", device="cpu")
        engine._pipeline = MagicMock()  # pre-inject so no real loading occurs
        first = engine._pipeline
        engine.load()
        assert engine._pipeline is first

    def test_unload_clears_pipeline(self):
        engine = StyleTransferEngine(device="cpu")
        engine._pipeline = MagicMock()
        engine.unload()
        assert engine._pipeline is None


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


class TestUtilities:
    def test_pil_to_numpy_shape(self):
        img = _make_image(32, 48)
        arr = _pil_to_numpy(img)
        assert arr.shape == (48, 32, 3)

    def test_numpy_to_pil_round_trip(self):
        import numpy as np

        arr = np.zeros((32, 32, 3), dtype=np.uint8)
        arr[:, :, 0] = 200
        img = _numpy_to_pil(arr)
        assert isinstance(img, Image.Image)
        assert img.size == (32, 32)
