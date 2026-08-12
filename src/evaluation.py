"""Evaluation metrics for reconstruction and contextual influence."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import ExperimentConfig
from .models import Autoencoder
from .utils import add_noise, generate_context


@torch.inference_mode()
def reconstruction_mse(
    model: Autoencoder,
    loader: DataLoader,
    config: ExperimentConfig,
    device: torch.device,
    use_context: bool = False,
    max_batches: int | None = None,
) -> float:
    """Calculate mean reconstruction error with either noisy or active context."""

    model.eval()
    total, batches = 0.0, 0
    for images, _ in loader:
        images = images.to(device)
        context = generate_context(
            len(images),
            signal_level=config.context_signal if use_context else None,
            noise_level=config.context_noise,
            category_id=config.context_channel,
            context_dim=config.context_dim,
            device=device,
        )
        reconstructed = model(add_noise(images, config.image_noise), context)
        total += nn.functional.mse_loss(reconstructed, images).item()
        batches += 1
        if max_batches is not None and batches >= max_batches:
            break
    if batches == 0:
        raise ValueError("evaluation loader contained no batches")
    return total / batches
