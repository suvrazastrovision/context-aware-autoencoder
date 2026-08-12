"""Train a baseline and context association from the command line."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ExperimentConfig
from src.data import create_mnist_loaders
from src.models import Autoencoder, count_parameters
from src.training import train_experiment
from src.utils import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs/default")
    parser.add_argument("--baseline-epochs", type=int, default=10)
    parser.add_argument("--association-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--target-digit", type=int, default=3)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-batches", type=int, default=None, help="Useful for smoke tests")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig(
        baseline_epochs=args.baseline_epochs,
        association_epochs=args.association_epochs,
        batch_size=args.batch_size,
        target_digit=args.target_digit,
    )
    config.validate()
    seed_everything(config.seed)
    device_name = (
        "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    )
    if device_name == "auto":
        device_name = "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device(device_name)

    train_loader, _ = create_mnist_loaders(batch_size=config.batch_size)
    model = Autoencoder(config.capacity, config.latent_dims, config.context_dim)
    print(json.dumps({"device": str(device), "parameters": count_parameters(model), **asdict(config)}, indent=2))
    train_experiment(model, train_loader, config, device, args.output_dir, args.max_batches)
    print(f"artifacts saved to {args.output_dir}")


if __name__ == "__main__":
    main()
