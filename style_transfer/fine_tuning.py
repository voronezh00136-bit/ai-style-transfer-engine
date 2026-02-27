"""
style_transfer/fine_tuning.py
-----------------------------
Style fine-tuning utilities.

``StyleFineTuner`` wraps Hugging Face PEFT / LoRA APIs to inject style-specific
adapter weights into a Stable Diffusion UNet so that the engine can reproduce
a particular artistic style (e.g. "Disney animation" or "Van Gogh brushwork")
without full model retraining.

Workflow
~~~~~~~~
1.  Instantiate ``StyleFineTuner`` with a base Stable Diffusion model ID.
2.  Call ``prepare(style_images, style_name)`` to build a training dataset.
3.  Call ``train(output_dir)`` to run LoRA fine-tuning and save the adapter.
4.  Call ``apply(pipeline, adapter_path)`` to inject the adapter into any
    ``StableDiffusionImg2ImgPipeline``.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Sequence

import torch
from PIL import Image

logger = logging.getLogger(__name__)


class StyleFineTuner:
    """Fine-tune a Stable Diffusion model to capture a specific artistic style.

    Uses Low-Rank Adaptation (LoRA) to efficiently adapt only a small subset
    of model parameters, keeping the base weights frozen.

    Args:
        base_model_id: Hugging Face model ID of the Stable Diffusion checkpoint
            to fine-tune.
        device: PyTorch device string.  Defaults to CUDA when available.
        dtype: Floating-point precision.  Defaults to ``float16`` on CUDA.
        lora_rank: Rank of the LoRA decomposition (higher = more capacity).
        lora_alpha: LoRA scaling factor.

    Example::

        from style_transfer import StyleFineTuner
        from PIL import Image

        tuner = StyleFineTuner()
        style_imgs = [Image.open(p) for p in ["van_gogh_1.jpg", "van_gogh_2.jpg"]]
        tuner.prepare(style_imgs, style_name="van_gogh")
        tuner.train("./adapters/van_gogh")
    """

    def __init__(
        self,
        base_model_id: str = "runwayml/stable-diffusion-v1-5",
        device: str | None = None,
        dtype: torch.dtype | None = None,
        lora_rank: int = 4,
        lora_alpha: int = 32,
    ) -> None:
        self.base_model_id = base_model_id
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.dtype = dtype or (
            torch.float16 if self.device.type == "cuda" else torch.float32
        )
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha

        self._style_name: str | None = None
        self._dataset: list[Image.Image] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def prepare(
        self,
        style_images: Sequence[Image.Image],
        style_name: str,
        resolution: int = 512,
    ) -> "StyleFineTuner":
        """Register a collection of style reference images for training.

        Images are centre-cropped and resized to ``resolution × resolution``.

        Args:
            style_images: Reference images exemplifying the target artistic style.
            style_name: Identifier used as the textual trigger token during training.
            resolution: Training image resolution (must be divisible by 8).

        Returns:
            ``self`` for method chaining.

        Raises:
            ValueError: When ``style_images`` is empty or ``resolution`` is not
                divisible by 8.
        """
        if not style_images:
            raise ValueError("At least one style image is required.")
        if resolution % 8 != 0:
            raise ValueError(f"resolution must be divisible by 8, got {resolution}.")

        self._style_name = style_name
        self._dataset = [self._preprocess(img, resolution) for img in style_images]
        logger.info(
            "Prepared %d style images for '%s' at %dx%d.",
            len(self._dataset),
            style_name,
            resolution,
            resolution,
        )
        return self

    def train(
        self,
        output_dir: str | os.PathLike,
        num_epochs: int = 1,
        learning_rate: float = 1e-4,
        batch_size: int = 1,
    ) -> Path:
        """Fine-tune the UNet with LoRA adapters on the prepared style images.

        Args:
            output_dir: Directory in which to save the LoRA adapter weights.
            num_epochs: Number of full passes over the style dataset.
            learning_rate: AdamW learning rate.
            batch_size: Mini-batch size (keep at 1 on consumer GPUs).

        Returns:
            ``Path`` to the saved adapter directory.

        Raises:
            RuntimeError: If ``prepare`` has not been called first.
        """
        if not self._dataset:
            raise RuntimeError("Call prepare() with style images before train().")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Starting LoRA fine-tuning for style '%s' — epochs=%d  lr=%g",
            self._style_name,
            num_epochs,
            learning_rate,
        )

        unet, noise_scheduler, vae, text_encoder, tokenizer = self._load_base_components()
        unet = self._inject_lora(unet)

        optimizer = torch.optim.AdamW(
            [p for p in unet.parameters() if p.requires_grad],
            lr=learning_rate,
        )

        unet.train()
        for epoch in range(num_epochs):
            epoch_loss = self._run_epoch(
                unet=unet,
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                noise_scheduler=noise_scheduler,
                optimizer=optimizer,
                batch_size=batch_size,
            )
            logger.info("Epoch %d/%d — loss: %.4f", epoch + 1, num_epochs, epoch_loss)

        adapter_path = self._save_adapter(unet, output_dir)
        logger.info("LoRA adapter saved to '%s'.", adapter_path)
        return adapter_path

    @staticmethod
    def apply(pipeline, adapter_path: str | os.PathLike) -> None:
        """Load a previously saved LoRA adapter into ``pipeline``.

        Args:
            pipeline: A ``StableDiffusionImg2ImgPipeline`` (or compatible)
                instance whose UNet will receive the adapter weights.
            adapter_path: Path to the directory produced by ``train``.

        Raises:
            FileNotFoundError: If ``adapter_path`` does not exist.
        """
        adapter_path = Path(adapter_path)
        if not adapter_path.exists():
            raise FileNotFoundError(f"Adapter directory not found: {adapter_path}")

        logger.info("Applying LoRA adapter from '%s'.", adapter_path)
        pipeline.unet.load_attn_procs(str(adapter_path))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_base_components(self):
        from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
        from transformers import CLIPTextModel, CLIPTokenizer

        logger.info("Loading base model components from '%s'.", self.base_model_id)
        tokenizer = CLIPTokenizer.from_pretrained(self.base_model_id, subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(
            self.base_model_id, subfolder="text_encoder", torch_dtype=self.dtype
        ).to(self.device)
        vae = AutoencoderKL.from_pretrained(
            self.base_model_id, subfolder="vae", torch_dtype=self.dtype
        ).to(self.device)
        unet = UNet2DConditionModel.from_pretrained(
            self.base_model_id, subfolder="unet", torch_dtype=self.dtype
        ).to(self.device)
        noise_scheduler = DDPMScheduler.from_pretrained(
            self.base_model_id, subfolder="scheduler"
        )

        # Freeze everything except the UNet (which gets LoRA adapters).
        vae.requires_grad_(False)
        text_encoder.requires_grad_(False)

        return unet, noise_scheduler, vae, text_encoder, tokenizer

    def _inject_lora(self, unet):
        """Replace cross-attention projections with LoRA-adapted versions."""
        from diffusers.loaders import AttnProcsLayers
        from diffusers.models.attention_processor import LoRAAttnProcessor

        lora_attn_procs = {}
        for name, attn_proc in unet.attn_processors.items():
            cross_attention_dim = (
                None
                if name.endswith("attn1.processor")
                else unet.config.cross_attention_dim
            )
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            else:
                hidden_size = unet.config.block_out_channels[0]

            lora_attn_procs[name] = LoRAAttnProcessor(
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim,
                rank=self.lora_rank,
            )

        unet.set_attn_processor(lora_attn_procs)
        lora_layers = AttnProcsLayers(unet.attn_processors)
        # Only LoRA parameters require gradients.
        unet.requires_grad_(False)
        lora_layers.requires_grad_(True)
        return unet

    def _run_epoch(
        self, *, unet, vae, text_encoder, tokenizer, noise_scheduler, optimizer, batch_size
    ) -> float:
        import random

        total_loss = 0.0
        images = self._dataset[:]
        random.shuffle(images)

        for i in range(0, len(images), batch_size):
            batch_images = images[i : i + batch_size]
            loss = self._training_step(
                unet=unet,
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                noise_scheduler=noise_scheduler,
                optimizer=optimizer,
                batch_images=batch_images,
            )
            total_loss += loss

        return total_loss / max(len(images), 1)

    def _training_step(
        self,
        *,
        unet,
        vae,
        text_encoder,
        tokenizer,
        noise_scheduler,
        optimizer,
        batch_images: list[Image.Image],
    ) -> float:
        import torchvision.transforms as T

        to_tensor = T.Compose([T.ToTensor(), T.Normalize([0.5], [0.5])])
        pixel_values = torch.stack([to_tensor(img) for img in batch_images]).to(
            device=self.device, dtype=self.dtype
        )

        # Encode images to latent space.
        with torch.no_grad():
            latents = vae.encode(pixel_values).latent_dist.sample() * vae.config.scaling_factor

        # Sample noise and timesteps.
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps, (bsz,), device=self.device
        ).long()
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        # Encode text conditioning.
        prompt = f"a painting in the style of {self._style_name}"
        text_inputs = tokenizer(
            [prompt] * bsz,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            encoder_hidden_states = text_encoder(text_inputs.input_ids)[0]

        # Forward pass.
        noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
        loss = torch.nn.functional.mse_loss(noise_pred.float(), noise.float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()

    @staticmethod
    def _save_adapter(unet, output_dir: Path) -> Path:
        unet.save_attn_procs(str(output_dir))
        return output_dir

    @staticmethod
    def _preprocess(image: Image.Image, resolution: int) -> Image.Image:
        """Centre-crop to square, then resize."""
        img = image.convert("RGB")
        w, h = img.size
        min_dim = min(w, h)
        left = (w - min_dim) // 2
        top = (h - min_dim) // 2
        img = img.crop((left, top, left + min_dim, top + min_dim))
        return img.resize((resolution, resolution), Image.Resampling.LANCZOS)
