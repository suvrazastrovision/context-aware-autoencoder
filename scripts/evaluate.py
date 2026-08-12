"""Evaluate a trained context-aware autoencoder."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ExperimentConfig
from src.data import create_mnist_loaders
from src.evaluation import reconstruction_mse
from src.models import Autoencoder
from src.utils import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="outputs/default")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--max-batches", type=int, default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    config = ExperimentConfig.load(run_dir / "config.json")
    seed_everything(config.seed)
    device = torch.device(args.device)
    _, test_loader = create_mnist_loaders(batch_size=config.batch_size)
    model = Autoencoder(config.capacity, config.latent_dims, config.context_dim).to(device)
    model.load_state_dict(torch.load(run_dir / "association.pt", map_location=device, weights_only=True))

    without_context = reconstruction_mse(model, test_loader, config, device, False, args.max_batches)
    with_context = reconstruction_mse(model, test_loader, config, device, True, args.max_batches)
    print(f"reconstruction MSE without context: {without_context:.6f}")
    print(f"reconstruction MSE with context:    {with_context:.6f}")
    print(f"context delta:                      {with_context - without_context:+.6f}")


if __name__ == "__main__":
    main()
