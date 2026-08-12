"""Plotting helpers used by the portfolio notebooks."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from torch import Tensor


def save_reconstruction_grid(
    original: Tensor,
    noisy: Tensor,
    reconstructed: Tensor,
    path: str | Path,
    count: int = 8,
) -> None:
    """Save original, corrupted, and reconstructed samples in a three-row grid."""

    count = min(count, len(original))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = (original, noisy, reconstructed)
    labels = ("Original", "Noisy", "Reconstructed")
    figure, axes = plt.subplots(3, count, figsize=(1.6 * count, 5))
    for row, (batch, label) in enumerate(zip(rows, labels)):
        for column in range(count):
            axes[row, column].imshow(batch[column].detach().cpu().squeeze(), cmap="gray", vmin=-1, vmax=1)
            axes[row, column].axis("off")
            if column == 0:
                axes[row, column].set_ylabel(label)
    figure.suptitle("Denoising autoencoder reconstruction")
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.show()


def save_context_comparison(
    original: Tensor,
    without_context: Tensor,
    with_context: Tensor,
    path: str | Path,
    count: int = 8,
) -> None:
    """Save side-by-side reconstructions with inactive and active context."""

    count = min(count, len(original))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = (original, without_context, with_context)
    labels = ("Original", "No context", "Active context")
    figure, axes = plt.subplots(3, count, figsize=(1.6 * count, 5))
    for row, (batch, label) in enumerate(zip(rows, labels)):
        for column in range(count):
            axes[row, column].imshow(batch[column].detach().cpu().squeeze(), cmap="gray", vmin=-1, vmax=1)
            axes[row, column].axis("off")
            if column == 0:
                axes[row, column].set_ylabel(label)
    figure.suptitle("Effect of the learned context signal")
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.show()
