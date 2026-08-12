"""Download and verify the complete MNIST training and test dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import create_mnist_loaders


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Dataset root; torchvision creates an MNIST subfolder here.",
    )
    args = parser.parse_args()

    train_loader, test_loader = create_mnist_loaders(
        data_dir=args.data_dir, batch_size=128
    )
    dataset_path = Path(args.data_dir).resolve() / "MNIST" / "raw"
    print(f"MNIST folder:    {dataset_path}")
    print(f"Training images: {len(train_loader.dataset):,}")
    print(f"Test images:     {len(test_loader.dataset):,}")
    print(f"Total images:    {len(train_loader.dataset) + len(test_loader.dataset):,}")


if __name__ == "__main__":
    main()
