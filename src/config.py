"""Typed experiment configuration shared by training and evaluation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExperimentConfig:
    capacity: int = 16
    latent_dims: int = 32
    context_dim: int = 10
    context_channel: int = 2
    target_digit: int = 3
    context_signal: float = 1.0
    context_noise: float = 0.1
    image_noise: float = 0.5
    batch_size: int = 128
    learning_rate: float = 1e-3
    baseline_epochs: int = 10
    association_epochs: int = 10
    seed: int = 42

    def validate(self) -> None:
        if min(self.capacity, self.latent_dims, self.context_dim, self.batch_size) <= 0:
            raise ValueError("model dimensions and batch_size must be positive")
        if not 0 <= self.target_digit <= 9:
            raise ValueError("target_digit must be between 0 and 9")
        if not 0 <= self.context_channel < self.context_dim:
            raise ValueError("context_channel must be smaller than context_dim")
        if self.baseline_epochs < 0 or self.association_epochs < 0:
            raise ValueError("epoch counts must be non-negative")

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> ExperimentConfig:
        config = cls(**json.loads(Path(path).read_text(encoding="utf-8")))
        config.validate()
        return config
