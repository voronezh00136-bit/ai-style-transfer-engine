"""
Tests for style_transfer.latent_explorer.LatentSpaceExplorer and _slerp.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch
from PIL import Image

from style_transfer.latent_explorer import LatentSpaceExplorer, _slerp

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_image(w: int = 64, h: int = 64) -> Image.Image:
    return Image.new("RGB", (w, h), color=(50, 100, 150))


def _make_explorer_with_mock_pipe() -> LatentSpaceExplorer:
    """Return a LatentSpaceExplorer with all heavy operations mocked out."""
    explorer = LatentSpaceExplorer(device="cpu", dtype=torch.float32)

    mock_pipe = MagicMock()

    # tokenizer
    mock_tokenizer = MagicMock()
    input_ids = torch.zeros((1, 77), dtype=torch.long)
    mock_tokenizer.return_value.to.return_value.input_ids = input_ids
    mock_tokenizer.model_max_length = 77
    mock_pipe.tokenizer = mock_tokenizer

    # text_encoder → returns (batch, seq, embed)
    mock_text_encoder = MagicMock()
    embedding = torch.randn(1, 77, 768)
    mock_text_encoder.return_value = [embedding]
    mock_pipe.text_encoder = mock_text_encoder

    # unet — noise_pred is a batched tensor (batch=2 for CFG) so .chunk(2) works.
    mock_unet = MagicMock()
    mock_unet.config.in_channels = 4
    mock_noise = MagicMock()
    mock_noise.sample = torch.randn(2, 4, 8, 8)
    mock_unet.return_value = mock_noise
    mock_pipe.unet = mock_unet

    # vae
    mock_vae = MagicMock()
    mock_vae.config.scaling_factor = 0.18215
    latent_dist = MagicMock()
    latent_dist.sample.return_value = torch.randn(1, 4, 8, 8)
    mock_vae.encode.return_value = MagicMock(latent_dist=latent_dist)
    # decode returns a tensor in [-1, 1] with the right shape
    decode_result = MagicMock()
    decode_result.sample = torch.randn(1, 3, 64, 64)
    mock_vae.decode.return_value = decode_result
    mock_pipe.vae = mock_vae

    # scheduler
    mock_scheduler = MagicMock()
    mock_scheduler.timesteps = [torch.tensor(t) for t in [999, 500, 1]]
    mock_scheduler.init_noise_sigma = 1.0
    step_result = MagicMock()
    step_result.prev_sample = torch.randn(1, 4, 8, 8)
    mock_scheduler.step.return_value = step_result
    mock_pipe.scheduler = mock_scheduler

    explorer._pipe = mock_pipe
    return explorer


# ---------------------------------------------------------------------------
# _slerp
# ---------------------------------------------------------------------------


class TestSlerp:
    def test_t0_returns_start(self):
        v0 = torch.tensor([1.0, 0.0, 0.0])
        v1 = torch.tensor([0.0, 1.0, 0.0])
        result = _slerp(v0, v1, 0.0)
        assert torch.allclose(result, v0.float(), atol=1e-5)

    def test_t1_returns_end(self):
        v0 = torch.tensor([1.0, 0.0, 0.0])
        v1 = torch.tensor([0.0, 1.0, 0.0])
        result = _slerp(v0, v1, 1.0)
        assert torch.allclose(result, v1.float(), atol=1e-5)

    def test_t_half_is_midpoint(self):
        v0 = torch.tensor([1.0, 0.0])
        v1 = torch.tensor([0.0, 1.0])
        result = _slerp(v0, v1, 0.5)
        # Mid-point on the unit circle at 45° → [√2/2, √2/2] (up to norm scaling).
        expected = torch.tensor([0.7071, 0.7071])
        # Normalise both to compare direction.
        result_norm = result / result.norm()
        expected_norm = expected / expected.norm()
        assert torch.allclose(result_norm.float(), expected_norm.float(), atol=1e-4)

    def test_parallel_vectors_fallback_to_lerp(self):
        v = torch.tensor([1.0, 0.0, 0.0])
        result = _slerp(v, v, 0.5)
        assert torch.allclose(result, v.float(), atol=1e-5)

    def test_preserves_dtype_float16(self):
        v0 = torch.tensor([1.0, 0.0], dtype=torch.float16)
        v1 = torch.tensor([0.0, 1.0], dtype=torch.float16)
        result = _slerp(v0, v1, 0.5)
        assert result.dtype == torch.float16

    def test_output_shape_matches_input(self):
        v0 = torch.randn(1, 77, 768)
        v1 = torch.randn(1, 77, 768)
        result = _slerp(v0, v1, 0.3)
        assert result.shape == v0.shape


# ---------------------------------------------------------------------------
# LatentSpaceExplorer initialisation
# ---------------------------------------------------------------------------


class TestInit:
    def test_defaults(self):
        explorer = LatentSpaceExplorer()
        assert explorer.model_id == "runwayml/stable-diffusion-v1-5"
        assert explorer._pipe is None

    def test_custom_device(self):
        explorer = LatentSpaceExplorer(device="cpu")
        assert explorer.device == torch.device("cpu")


# ---------------------------------------------------------------------------
# interpolate_styles()
# ---------------------------------------------------------------------------


class TestInterpolateStyles:
    def test_raises_if_steps_lt_2(self):
        explorer = _make_explorer_with_mock_pipe()
        with pytest.raises(ValueError, match="steps must be >= 2"):
            explorer.interpolate_styles("a", "b", steps=1)

    def test_returns_correct_number_of_frames(self):
        explorer = _make_explorer_with_mock_pipe()
        frames = explorer.interpolate_styles("Van Gogh", "Disney", steps=4)
        assert len(frames) == 4

    def test_all_frames_are_pil_images(self):
        explorer = _make_explorer_with_mock_pipe()
        frames = explorer.interpolate_styles("a", "b", steps=3)
        for f in frames:
            assert isinstance(f, Image.Image)

    def test_endpoints_are_included(self):
        explorer = _make_explorer_with_mock_pipe()
        # Just check we get exactly `steps` frames (endpoints included).
        frames = explorer.interpolate_styles("x", "y", steps=2)
        assert len(frames) == 2


# ---------------------------------------------------------------------------
# interpolate_prompts()
# ---------------------------------------------------------------------------


class TestInterpolatePrompts:
    def test_raises_with_single_prompt(self):
        explorer = _make_explorer_with_mock_pipe()
        with pytest.raises(ValueError, match="At least 2 prompts"):
            explorer.interpolate_prompts(["only one"])

    def test_correct_frame_count_two_prompts(self):
        explorer = _make_explorer_with_mock_pipe()
        # 2 anchors + steps_between intermediate per segment = steps_between + 2 total
        frames = explorer.interpolate_prompts(["a", "b"], steps_between=2)
        # segment: [a...b] inclusive = steps_between + 2 = 4
        assert len(frames) == 4

    def test_correct_frame_count_three_prompts(self):
        explorer = _make_explorer_with_mock_pipe()
        # [a...b] gives steps_between + 1 (last excluded) + [b...c] gives steps_between + 2
        steps_between = 2
        frames = explorer.interpolate_prompts(["a", "b", "c"], steps_between=steps_between)
        expected = (steps_between + 1) + (steps_between + 2)
        assert len(frames) == expected

    def test_all_frames_pil_images(self):
        explorer = _make_explorer_with_mock_pipe()
        frames = explorer.interpolate_prompts(["a", "b"], steps_between=1)
        for f in frames:
            assert isinstance(f, Image.Image)


# ---------------------------------------------------------------------------
# steer_latent()
# ---------------------------------------------------------------------------


class TestSteerLatent:
    def test_output_shape_preserved(self):
        explorer = LatentSpaceExplorer(device="cpu")
        latent = torch.randn(1, 4, 8, 8)
        direction = torch.randn(1, 4, 8, 8)
        result = explorer.steer_latent(latent, direction, magnitude=0.5)
        assert result.shape == latent.shape

    def test_magnitude_zero_returns_original(self):
        explorer = LatentSpaceExplorer(device="cpu")
        latent = torch.randn(1, 4, 8, 8)
        direction = torch.randn(1, 4, 8, 8)
        result = explorer.steer_latent(latent, direction, magnitude=0.0)
        assert torch.allclose(result, latent, atol=1e-6)

    def test_steered_differs_from_original(self):
        explorer = LatentSpaceExplorer(device="cpu")
        latent = torch.randn(1, 4, 8, 8)
        direction = torch.ones(1, 4, 8, 8)
        result = explorer.steer_latent(latent, direction, magnitude=1.0)
        assert not torch.allclose(result, latent)


# ---------------------------------------------------------------------------
# load() idempotence
# ---------------------------------------------------------------------------


class TestLoad:
    def test_load_is_idempotent(self):
        explorer = LatentSpaceExplorer(device="cpu")
        explorer._pipe = MagicMock()  # pre-inject
        first = explorer._pipe
        explorer.load()
        assert explorer._pipe is first
