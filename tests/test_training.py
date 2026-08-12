import torch
from torch.utils.data import DataLoader, TensorDataset

from src.config import ExperimentConfig
from src.models import Autoencoder
from src.training import train_experiment


def test_one_batch_training_creates_artifacts(tmp_path):
    images = torch.rand(4, 1, 28, 28) * 2 - 1
    labels = torch.tensor([0, 3, 5, 3])
    loader = DataLoader(TensorDataset(images, labels), batch_size=4)
    config = ExperimentConfig(
        capacity=2,
        latent_dims=4,
        context_dim=3,
        context_channel=1,
        baseline_epochs=1,
        association_epochs=1,
        batch_size=4,
    )
    model = Autoencoder(config.capacity, config.latent_dims, config.context_dim)
    history = train_experiment(
        model, loader, config, torch.device("cpu"), tmp_path, max_batches=1
    )
    assert len(history.baseline_loss) == 1
    assert len(history.association_loss) == 1
    assert (tmp_path / "baseline.pt").exists()
    assert (tmp_path / "association.pt").exists()
    assert (tmp_path / "config.json").exists()
    assert (tmp_path / "history.json").exists()
