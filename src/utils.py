"""Reproducibility, context generation, and image corruption helpers."""

from __future__ import annotations

import random

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F


def seed_everything(seed: int = 42) -> None:
    """Seed Python, NumPy, and PyTorch for repeatable experiments."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_context(
    batch_size: int,
    signal_level: float | None = None,
    noise_level: float = 0.1,
    category_id: int = 2,
    context_dim: int = 10,
    device: str | torch.device = "cpu",
) -> Tensor:
    """Create noisy context vectors and optionally activate one channel."""

    if batch_size <= 0 or context_dim <= 0:
        raise ValueError("batch_size and context_dim must be positive")
    if not 0 <= category_id < context_dim:
        raise ValueError("category_id must index a valid context channel")
    if noise_level < 0:
        raise ValueError("noise_level must be non-negative")

    context = noise_level * torch.randn(batch_size, context_dim, device=device)
    if signal_level is not None:
        context[:, category_id] = signal_level
    return context


def add_noise(
    images: Tensor,
    noise_level: float = 0.5,
    kernel_size: int = 5,
    sigma: float = 2.0,
    correlated: bool = False,
    uncorrelated: bool = True,
) -> Tensor:
    """Add white and/or Gaussian-smoothed noise and clamp images to [-1, 1]."""

    if images.ndim == 3:
        images = images.unsqueeze(1)
    if images.ndim != 4:
        raise ValueError("images must have shape (batch, channels, height, width)")
    if noise_level < 0:
        raise ValueError("noise_level must be non-negative")
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")
    if sigma <= 0:
        raise ValueError("sigma must be positive")

    noise = torch.zeros_like(images)
    if uncorrelated:
        noise.add_(noise_level * torch.randn_like(images))

    if correlated:
        smooth_noise = torch.randn_like(images)
        grid = torch.arange(kernel_size, dtype=images.dtype, device=images.device)
        grid -= (kernel_size - 1) / 2
        kernel_1d = torch.exp(-0.5 * grid.square() / sigma**2)
        kernel_1d /= kernel_1d.sum()
        kernel = (kernel_1d[:, None] * kernel_1d[None, :]).expand(
            images.shape[1], 1, kernel_size, kernel_size
        )
        smooth_noise = F.conv2d(
            smooth_noise, kernel, padding=kernel_size // 2, groups=images.shape[1]
        )
        std = smooth_noise.std().clamp_min(torch.finfo(images.dtype).eps)
        noise.add_(noise_level * smooth_noise / std)

    return (images + noise).clamp(-1, 1)
