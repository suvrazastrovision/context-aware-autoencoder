"""Neural-network components for context-aware MNIST reconstruction."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class Encoder(nn.Module):
    """Encode a 28x28 grayscale image and an auxiliary context vector."""

    def __init__(self, capacity: int = 16, latent_dims: int = 32, context_dim: int = 10):
        super().__init__()
        if min(capacity, latent_dims, context_dim) <= 0:
            raise ValueError("capacity, latent_dims, and context_dim must be positive")

        self.context_dim = context_dim
        self.conv1 = nn.Conv2d(1, capacity, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(capacity, capacity * 2, kernel_size=4, stride=2, padding=1)
        self.fc1 = nn.Linear(capacity * 2 * 7 * 7, latent_dims)
        self.fc_context = nn.Linear(context_dim, latent_dims)

    def forward(self, image: Tensor, context: Tensor) -> Tensor:
        if image.ndim != 4 or image.shape[1:] != (1, 28, 28):
            raise ValueError("image must have shape (batch, 1, 28, 28)")
        if context.ndim != 2 or context.shape != (image.shape[0], self.context_dim):
            raise ValueError(f"context must have shape (batch, {self.context_dim})")

        features = F.relu(self.conv1(image))
        features = F.relu(self.conv2(features))
        features = features.flatten(start_dim=1)
        return self.fc1(features) + self.fc_context(context)


class Decoder(nn.Module):
    """Decode a latent vector into a normalized 28x28 grayscale image."""

    def __init__(self, capacity: int = 16, latent_dims: int = 32, context_dim: int = 10):
        super().__init__()
        self.capacity = capacity
        self.fc = nn.Linear(latent_dims, capacity * 2 * 7 * 7)
        self.conv2 = nn.ConvTranspose2d(
            capacity * 2, capacity, kernel_size=4, stride=2, padding=1
        )
        self.conv1 = nn.ConvTranspose2d(
            capacity, 1, kernel_size=4, stride=2, padding=1
        )

    def forward(self, latent: Tensor, context: Tensor | None = None) -> Tensor:
        # ``context`` remains in the signature for compatibility. The encoder has
        # already incorporated it into ``latent``.
        features = self.fc(latent)
        features = features.view(latent.shape[0], self.capacity * 2, 7, 7)
        features = F.relu(self.conv2(features))
        return torch.tanh(self.conv1(features))


class Autoencoder(nn.Module):
    """Context-conditioned convolutional denoising autoencoder."""

    def __init__(self, capacity: int = 16, latent_dims: int = 32, context_dim: int = 10):
        super().__init__()
        self.context_dim = context_dim
        self.encoder = Encoder(capacity, latent_dims, context_dim)
        self.decoder = Decoder(capacity, latent_dims, context_dim)

    def forward(self, image: Tensor, context: Tensor) -> Tensor:
        return self.decoder(self.encoder(image, context), context)


def freeze_except_context(model: nn.Module) -> None:
    """Freeze a model except for the encoder's context projection."""

    for name, parameter in model.named_parameters():
        parameter.requires_grad = "fc_context" in name


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """Return the number of model parameters."""

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if not trainable_only or parameter.requires_grad
    )
