"""Reusable baseline and context-association training loops."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import ExperimentConfig
from .models import Autoencoder, freeze_except_context
from .utils import add_noise, generate_context


@dataclass
class TrainingHistory:
    baseline_loss: list[float]
    association_loss: list[float]


def _train_epoch(
    model: Autoencoder,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    config: ExperimentConfig,
    device: torch.device,
    associated: bool,
    max_batches: int | None = None,
) -> float:
    model.train()
    total_loss = 0.0
    batches = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        context = generate_context(
            len(images),
            noise_level=config.context_noise,
            category_id=config.context_channel,
            context_dim=config.context_dim,
            device=device,
        )
        if associated:
            target_mask = labels == config.target_digit
            context[target_mask, config.context_channel] = config.context_signal

        corrupted = add_noise(images, noise_level=config.image_noise)
        loss = nn.functional.mse_loss(model(corrupted, context), images)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        batches += 1
        if max_batches is not None and batches >= max_batches:
            break

    if batches == 0:
        raise ValueError("training loader contained no batches")
    return total_loss / batches


def train_experiment(
    model: Autoencoder,
    train_loader: DataLoader,
    config: ExperimentConfig,
    device: torch.device,
    output_dir: str | Path = "outputs",
    max_batches: int | None = None,
) -> TrainingHistory:
    """Train the baseline, freeze it, then learn the context association."""

    config.validate()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    baseline_loss = []
    for epoch in range(config.baseline_epochs):
        value = _train_epoch(
            model, train_loader, optimizer, config, device, False, max_batches
        )
        baseline_loss.append(value)
        print(f"baseline {epoch + 1:03d}/{config.baseline_epochs}: loss={value:.6f}")

    torch.save(model.state_dict(), output_dir / "baseline.pt")

    freeze_except_context(model)
    optimizer = torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config.learning_rate,
    )
    association_loss = []
    for epoch in range(config.association_epochs):
        value = _train_epoch(
            model, train_loader, optimizer, config, device, True, max_batches
        )
        association_loss.append(value)
        print(
            f"association {epoch + 1:03d}/{config.association_epochs}: "
            f"loss={value:.6f}"
        )

    torch.save(model.state_dict(), output_dir / "association.pt")
    config.save(output_dir / "config.json")
    history = TrainingHistory(baseline_loss, association_loss)
    (output_dir / "history.json").write_text(
        json.dumps(asdict(history), indent=2) + "\n", encoding="utf-8"
    )
    return history
