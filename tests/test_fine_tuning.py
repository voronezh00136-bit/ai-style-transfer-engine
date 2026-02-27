"""
Tests for style_transfer.fine_tuning.StyleFineTuner.

Heavy model loading is mocked; training math is exercised with tiny tensors.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch
from PIL import Image

from style_transfer.fine_tuning import StyleFineTuner

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_image(w: int = 64, h: int = 64) -> Image.Image:
    return Image.new("RGB", (w, h), color=(100, 150, 200))


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


class TestStyleFineTunerInit:
    def test_defaults(self):
        tuner = StyleFineTuner()
        assert tuner.lora_rank == 4
        assert tuner.lora_alpha == 32
        assert tuner._dataset == []
        assert tuner._style_name is None

    def test_custom_rank(self):
        tuner = StyleFineTuner(lora_rank=8)
        assert tuner.lora_rank == 8


# ---------------------------------------------------------------------------
# prepare()
# ---------------------------------------------------------------------------


class TestPrepare:
    def test_empty_images_raises(self):
        tuner = StyleFineTuner()
        with pytest.raises(ValueError, match="At least one style image"):
            tuner.prepare([], "van_gogh")

    def test_bad_resolution_raises(self):
        tuner = StyleFineTuner()
        imgs = [_make_image()]
        with pytest.raises(ValueError, match="divisible by 8"):
            tuner.prepare(imgs, "test", resolution=33)

    def test_images_preprocessed_to_resolution(self):
        tuner = StyleFineTuner()
        imgs = [_make_image(200, 300), _make_image(100, 100)]
        tuner.prepare(imgs, "monet", resolution=64)
        assert len(tuner._dataset) == 2
        for img in tuner._dataset:
            assert img.size == (64, 64)
            assert img.mode == "RGB"

    def test_style_name_stored(self):
        tuner = StyleFineTuner()
        tuner.prepare([_make_image()], "disney")
        assert tuner._style_name == "disney"

    def test_returns_self(self):
        tuner = StyleFineTuner()
        result = tuner.prepare([_make_image()], "test")
        assert result is tuner


# ---------------------------------------------------------------------------
# _preprocess()
# ---------------------------------------------------------------------------


class TestPreprocess:
    def test_square_crop_result(self):
        img = _make_image(200, 100)
        result = StyleFineTuner._preprocess(img, resolution=64)
        assert result.size == (64, 64)

    def test_converts_rgba_to_rgb(self):
        img = Image.new("RGBA", (64, 64))
        result = StyleFineTuner._preprocess(img, resolution=64)
        assert result.mode == "RGB"

    def test_already_square_does_not_distort(self):
        img = _make_image(128, 128)
        result = StyleFineTuner._preprocess(img, resolution=64)
        assert result.size == (64, 64)


# ---------------------------------------------------------------------------
# train() — with mocked model components
# ---------------------------------------------------------------------------


class TestTrain:
    def test_train_without_prepare_raises(self, tmp_path):
        tuner = StyleFineTuner()
        with pytest.raises(RuntimeError, match="Call prepare"):
            tuner.train(tmp_path)

    def test_train_returns_path(self, tmp_path):
        tuner = StyleFineTuner(device="cpu", dtype=torch.float32)
        tuner.prepare([_make_image()], "van_gogh", resolution=64)

        # Patch all heavy components.
        with patch.object(tuner, "_load_base_components") as mock_load, \
             patch.object(tuner, "_inject_lora") as mock_lora, \
             patch.object(tuner, "_run_epoch", return_value=0.5) as mock_epoch, \
             patch.object(StyleFineTuner, "_save_adapter", return_value=tmp_path):

            mock_unet = MagicMock()
            mock_unet.parameters.return_value = [torch.zeros(1, requires_grad=True)]
            mock_load.return_value = (
                mock_unet,    # unet
                MagicMock(),  # noise_scheduler
                MagicMock(),  # vae
                MagicMock(),  # text_encoder
                MagicMock(),  # tokenizer
            )
            mock_lora.return_value = mock_unet

            result = tuner.train(tmp_path, num_epochs=1)

        assert result == tmp_path
        mock_epoch.assert_called_once()

    def test_train_creates_output_dir(self, tmp_path):
        output = tmp_path / "new_adapter_dir"
        tuner = StyleFineTuner(device="cpu", dtype=torch.float32)
        tuner.prepare([_make_image()], "test", resolution=64)

        with patch.object(tuner, "_load_base_components") as mock_load, \
             patch.object(tuner, "_inject_lora") as mock_lora, \
             patch.object(tuner, "_run_epoch", return_value=0.0), \
             patch.object(StyleFineTuner, "_save_adapter", return_value=output):

            mock_unet = MagicMock()
            mock_unet.parameters.return_value = [torch.zeros(1, requires_grad=True)]
            mock_load.return_value = (mock_unet, MagicMock(), MagicMock(), MagicMock(), MagicMock())
            mock_lora.return_value = mock_unet

            tuner.train(output, num_epochs=1)

        assert output.exists()


# ---------------------------------------------------------------------------
# apply()
# ---------------------------------------------------------------------------


class TestApply:
    def test_apply_raises_if_not_found(self, tmp_path):
        missing = tmp_path / "nonexistent"
        pipeline = MagicMock()
        with pytest.raises(FileNotFoundError):
            StyleFineTuner.apply(pipeline, missing)

    def test_apply_calls_load_attn_procs(self, tmp_path):
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        pipeline = MagicMock()
        StyleFineTuner.apply(pipeline, adapter_dir)
        pipeline.unet.load_attn_procs.assert_called_once_with(str(adapter_dir))
