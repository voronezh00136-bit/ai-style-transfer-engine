"""
style_transfer/latent_explorer.py
----------------------------------
Latent-space exploration and interpolation utilities.

``LatentSpaceExplorer`` provides tools for navigating the latent space of a
Stable Diffusion model, enabling:

* **Style interpolation** — blend two style prompts at arbitrary ratios to
  produce a smooth aesthetic transition.
* **Prompt interpolation** — create a multi-frame journey between two text
  prompts (useful for animations or style-transition sequences).
* **Latent arithmetic** — add or subtract directional vectors to steer the
  content of an encoded image without modifying the diffusion schedule.

The explorer operates on *prompt embeddings* (text encoder output tensors)
rather than image pixels, which gives smooth, semantically coherent transitions
that are impossible to achieve by blending pixel values directly.
"""

from __future__ import annotations

import logging
from typing import Sequence

import torch
from PIL import Image

logger = logging.getLogger(__name__)


class LatentSpaceExplorer:
    """Explore and interpolate between styles in the latent space of a diffusion model.

    Args:
        model_id: Hugging Face model ID of the Stable Diffusion pipeline.
        device: PyTorch device string.  Defaults to CUDA when available.
        dtype: Floating-point precision.  Defaults to ``float16`` on CUDA.

    Example::

        from style_transfer import LatentSpaceExplorer

        explorer = LatentSpaceExplorer()
        frames = explorer.interpolate_styles(
            start_prompt="Van Gogh starry night",
            end_prompt="Disney animation style",
            steps=8,
            seed=42,
        )
        for i, frame in enumerate(frames):
            frame.save(f"frame_{i:02d}.png")
    """

    def __init__(
        self,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        device: str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        self.model_id = model_id
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.dtype = dtype or (
            torch.float16 if self.device.type == "cuda" else torch.float32
        )
        self._pipe = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> "LatentSpaceExplorer":
        """Load the Stable Diffusion pipeline.

        Called automatically on the first exploration call.

        Returns:
            ``self`` for method chaining.
        """
        if self._pipe is not None:
            return self
        from diffusers import StableDiffusionPipeline

        logger.info("Loading Stable Diffusion pipeline for latent exploration: %s", self.model_id)
        self._pipe = StableDiffusionPipeline.from_pretrained(
            self.model_id,
            torch_dtype=self.dtype,
            safety_checker=None,
        ).to(self.device)
        return self

    def interpolate_styles(
        self,
        start_prompt: str,
        end_prompt: str,
        steps: int = 8,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 30,
        height: int = 512,
        width: int = 512,
        seed: int | None = None,
    ) -> list[Image.Image]:
        """Generate a sequence of images that interpolate between two style prompts.

        Uses *spherical linear interpolation* (slerp) in the prompt-embedding
        space to produce a perceptually uniform transition.

        Args:
            start_prompt: Text description of the starting style.
            end_prompt: Text description of the ending style.
            steps: Number of frames in the interpolation (including endpoints).
            guidance_scale: Classifier-free guidance scale.
            num_inference_steps: Denoising steps per frame.
            height: Output image height in pixels.
            width: Output image width in pixels.
            seed: Optional random seed for reproducibility.

        Returns:
            List of ``steps`` PIL images.

        Raises:
            ValueError: If ``steps`` is less than 2.
        """
        if steps < 2:
            raise ValueError(f"steps must be >= 2, got {steps}.")

        self.load()

        start_emb = self._encode_prompt(start_prompt)
        end_emb = self._encode_prompt(end_prompt)

        alphas = [i / (steps - 1) for i in range(steps)]
        frames: list[Image.Image] = []
        for alpha in alphas:
            interp_emb = _slerp(start_emb, end_emb, alpha)
            image = self._decode_embedding(
                interp_emb,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                height=height,
                width=width,
                seed=seed,
            )
            frames.append(image)
            logger.debug("Interpolation step α=%.3f done.", alpha)

        return frames

    def interpolate_prompts(
        self,
        prompts: Sequence[str],
        steps_between: int = 4,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 30,
        height: int = 512,
        width: int = 512,
        seed: int | None = None,
    ) -> list[Image.Image]:
        """Interpolate through an ordered list of prompts.

        Creates a smooth visual journey through all provided prompts by slerp-
        interpolating consecutive pairs.

        Args:
            prompts: Ordered list of text prompts to traverse.
            steps_between: Number of intermediate frames between each adjacent
                pair of prompts (not including the anchor frames themselves).
            guidance_scale: Classifier-free guidance scale.
            num_inference_steps: Denoising steps per frame.
            height: Output image height.
            width: Output image width.
            seed: Optional random seed.

        Returns:
            List of PIL images representing the full journey.

        Raises:
            ValueError: If fewer than 2 prompts are provided.
        """
        if len(prompts) < 2:
            raise ValueError("At least 2 prompts are required for interpolation.")

        self.load()

        frames: list[Image.Image] = []
        embeddings = [self._encode_prompt(p) for p in prompts]

        for i in range(len(embeddings) - 1):
            total_steps = steps_between + 2  # include both endpoints
            segment_alphas = [j / (total_steps - 1) for j in range(total_steps)]
            # Skip the last alpha for all segments except the final one to
            # avoid duplicating anchor frames.
            if i < len(embeddings) - 2:
                segment_alphas = segment_alphas[:-1]

            for alpha in segment_alphas:
                interp_emb = _slerp(embeddings[i], embeddings[i + 1], alpha)
                image = self._decode_embedding(
                    interp_emb,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    height=height,
                    width=width,
                    seed=seed,
                )
                frames.append(image)

        return frames

    def embed_image(self, image: Image.Image) -> torch.Tensor:
        """Encode a PIL image into the model's latent space.

        Args:
            image: Input PIL image (converted to RGB internally).

        Returns:
            Latent tensor of shape ``(1, C, H/8, W/8)``.
        """
        self.load()
        import torchvision.transforms as T

        transform = T.Compose([
            T.Resize((512, 512)),
            T.ToTensor(),
            T.Normalize([0.5], [0.5]),
        ])
        pixel_values = transform(image.convert("RGB")).unsqueeze(0).to(
            device=self.device, dtype=self.dtype
        )
        with torch.inference_mode():
            latent = self._pipe.vae.encode(pixel_values).latent_dist.sample()
            latent = latent * self._pipe.vae.config.scaling_factor
        return latent

    def steer_latent(
        self,
        latent: torch.Tensor,
        direction: torch.Tensor,
        magnitude: float = 1.0,
    ) -> torch.Tensor:
        """Apply a directional offset to a latent vector.

        This enables *latent arithmetic* — e.g. moving towards a style
        direction obtained by subtracting two prompt embeddings.

        Args:
            latent: Source latent tensor (e.g. from ``embed_image``).
            direction: Direction vector in the same latent space.
            magnitude: Scalar multiplier for the direction vector.

        Returns:
            Steered latent tensor.
        """
        direction = direction.to(dtype=latent.dtype, device=latent.device)
        direction_norm = direction / (direction.norm() + 1e-8)
        return latent + magnitude * direction_norm

    def decode_latent(self, latent: torch.Tensor) -> Image.Image:
        """Decode a latent tensor back to a PIL image.

        Args:
            latent: Latent tensor of shape ``(1, C, H/8, W/8)``.

        Returns:
            Decoded PIL image.
        """
        self.load()
        import torchvision.transforms.functional as TF

        with torch.inference_mode():
            image_tensor = self._pipe.vae.decode(
                latent / self._pipe.vae.config.scaling_factor
            ).sample
        image_tensor = (image_tensor / 2 + 0.5).clamp(0, 1).squeeze(0).float().cpu()
        return TF.to_pil_image(image_tensor)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode_prompt(self, prompt: str) -> torch.Tensor:
        """Return a ``(1, seq_len, embed_dim)`` text embedding for ``prompt``."""
        text_inputs = self._pipe.tokenizer(
            prompt,
            padding="max_length",
            max_length=self._pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        with torch.inference_mode():
            embedding = self._pipe.text_encoder(text_inputs.input_ids)[0]
        return embedding.to(dtype=self.dtype)

    def _decode_embedding(
        self,
        embedding: torch.Tensor,
        *,
        guidance_scale: float,
        num_inference_steps: int,
        height: int,
        width: int,
        seed: int | None,
    ) -> Image.Image:
        """Run the full denoising pass from a prompt embedding to a PIL image."""
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        # Unconditional embedding for CFG.
        uncond_inputs = self._pipe.tokenizer(
            "",
            padding="max_length",
            max_length=self._pipe.tokenizer.model_max_length,
            return_tensors="pt",
        ).to(self.device)
        with torch.inference_mode():
            uncond_embedding = self._pipe.text_encoder(uncond_inputs.input_ids)[0].to(
                dtype=self.dtype
            )

        prompt_embeds = torch.cat([uncond_embedding, embedding])

        self._pipe.scheduler.set_timesteps(num_inference_steps)
        latents = torch.randn(
            (1, self._pipe.unet.config.in_channels, height // 8, width // 8),
            generator=generator,
            device=self.device,
            dtype=self.dtype,
        )
        latents = latents * self._pipe.scheduler.init_noise_sigma

        for t in self._pipe.scheduler.timesteps:
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self._pipe.scheduler.scale_model_input(latent_model_input, t)
            with torch.inference_mode():
                noise_pred = self._pipe.unet(
                    latent_model_input, t, encoder_hidden_states=prompt_embeds
                ).sample
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )
            latents = self._pipe.scheduler.step(noise_pred, t, latents).prev_sample

        return self.decode_latent(latents)


# ---------------------------------------------------------------------------
# Spherical linear interpolation (slerp)
# ---------------------------------------------------------------------------


def _slerp(v0: torch.Tensor, v1: torch.Tensor, t: float) -> torch.Tensor:
    """Spherical linear interpolation between two tensors.

    Args:
        v0: Start tensor.
        v1: End tensor (same shape as ``v0``).
        t: Interpolation factor in ``[0, 1]``.

    Returns:
        Interpolated tensor of the same shape.
    """
    v0_flat = v0.reshape(-1).float()
    v1_flat = v1.reshape(-1).float()

    dot = torch.dot(v0_flat / v0_flat.norm(), v1_flat / v1_flat.norm()).clamp(-1, 1)
    theta = torch.acos(dot)

    if theta.abs() < 1e-6:
        # Vectors are nearly parallel — fall back to linear interpolation.
        return ((1 - t) * v0 + t * v1).to(dtype=v0.dtype)

    sin_theta = torch.sin(theta)
    coeff0 = torch.sin((1 - t) * theta) / sin_theta
    coeff1 = torch.sin(t * theta) / sin_theta
    return (coeff0 * v0_flat + coeff1 * v1_flat).reshape(v0.shape).to(dtype=v0.dtype)
