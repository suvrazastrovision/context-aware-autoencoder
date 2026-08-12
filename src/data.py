"""MNIST data loading kept separate from model code."""

from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader


def create_mnist_loaders(
    data_dir: str | Path = "data",
    batch_size: int = 128,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """Download MNIST when needed and return deterministic train/test loaders."""

    try:
        from torchvision import transforms
        from torchvision.datasets import MNIST
    except ImportError as exc:
        raise RuntimeError(
            "torchvision is required for MNIST. Run: pip install -r requirements.txt"
        ) from exc

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    train_set = MNIST(data_dir, train=True, download=True, transform=transform)
    test_set = MNIST(data_dir, train=False, download=True, transform=transform)
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, test_loader
