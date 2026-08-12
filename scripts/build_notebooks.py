"""Generate the clean, ordered portfolio notebooks."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"
(ROOT / "assets").mkdir(parents=True, exist_ok=True)


def markdown(text: str):
    return nbf.v4.new_markdown_cell(dedent(text).strip())


def code(text: str):
    return nbf.v4.new_code_cell(dedent(text).strip())


def write(name: str, cells: list) -> None:
    notebook = nbf.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "AutoEncoder Portfolio",
                "language": "python",
                "name": "autoencoder-portfolio",
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
    )
    NOTEBOOKS.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, NOTEBOOKS / name)


COMMON = """
import os
from pathlib import Path
import sys

ROOT = Path.cwd().resolve()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".jupyter_runtime" / "matplotlib"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
"""


write(
    "01_data_and_architecture.ipynb",
    [
        markdown("""
        # 1. Data and architecture

        This notebook introduces the normalized MNIST input, generates controlled
        image corruption, and verifies the context-aware autoencoder's tensor flow.
        It performs no expensive training and can be run independently.
        """),
        code(COMMON),
        code("""
        import matplotlib.pyplot as plt
        import torch

        from src.config import ExperimentConfig
        from src.data import create_mnist_loaders
        from src.models import Autoencoder, count_parameters
        from src.utils import add_noise, generate_context, seed_everything

        config = ExperimentConfig()
        seed_everything(config.seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _, test_loader = create_mnist_loaders(ROOT / "data", config.batch_size)
        images, labels = next(iter(test_loader))
        images = images.to(device)
        print(f"Device: {device}")
        print(f"Dataset sizes: train=60,000, test=10,000")
        print(f"Image batch: {tuple(images.shape)}; value range: [{images.min():.1f}, {images.max():.1f}]")
        """),
        markdown("""
        ## Controlled corruption

        MNIST is normalized to `[-1, 1]`. Gaussian noise makes the reconstruction
        task ambiguous while preserving the original image as the learning target.
        """),
        code("""
        noisy = add_noise(images, noise_level=config.image_noise)
        figure, axes = plt.subplots(2, 8, figsize=(12, 3.5))
        for column in range(8):
            for row, batch in enumerate((images, noisy)):
                axes[row, column].imshow(batch[column].cpu().squeeze(), cmap="gray", vmin=-1, vmax=1)
                axes[row, column].axis("off")
        axes[0, 0].set_ylabel("Original")
        axes[1, 0].set_ylabel("Noisy")
        figure.suptitle("MNIST denoising inputs")
        figure.tight_layout()
        plt.show()
        """),
        markdown("""
        ## Context-conditioned model

        The image pathway produces a latent representation. A linear projection of
        the auxiliary context vector is added to it before decoding. Context therefore
        changes reconstruction through the latent representation.
        """),
        code("""
        model = Autoencoder(config.capacity, config.latent_dims, config.context_dim).to(device)
        context = generate_context(
            len(images), context_dim=config.context_dim,
            category_id=config.context_channel, device=device
        )
        with torch.inference_mode():
            latent = model.encoder(noisy, context)
            reconstructed = model.decoder(latent)
        print(model)
        print(f"Parameters: {count_parameters(model):,}")
        print(f"Latent shape: {tuple(latent.shape)}")
        print(f"Reconstruction shape: {tuple(reconstructed.shape)}")
        assert reconstructed.shape == images.shape
        """),
        markdown("""
        **Interpretation.** Each 28×28 image is compressed to a 32-dimensional
        vector and reconstructed to its original shape. At this point the network is
        untrained; the next notebook learns the visual and contextual mappings.
        """),
    ],
)


write(
    "02_training.ipynb",
    [
        markdown("""
        # 2. Two-stage training

        First, the complete network learns denoising. Second, the visual pathway is
        frozen and only the context projection learns an association with digit `3`.

        `QUICK_RUN=True` keeps this notebook practical on CPU. Set it to `False` for
        the portfolio experiment (10 epochs per phase over all batches).
        """),
        code(COMMON),
        code("""
        import json
        import matplotlib.pyplot as plt
        import torch

        from src.config import ExperimentConfig
        from src.data import create_mnist_loaders
        from src.models import Autoencoder, count_parameters
        from src.training import train_experiment
        from src.utils import add_noise, generate_context, seed_everything
        from src.visualization import save_reconstruction_grid

        QUICK_RUN = True
        config = ExperimentConfig(
            baseline_epochs=1 if QUICK_RUN else 10,
            association_epochs=1 if QUICK_RUN else 10,
        )
        max_batches = 20 if QUICK_RUN else None
        output_dir = ROOT / "outputs" / ("notebook_quick" if QUICK_RUN else "portfolio")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        seed_everything(config.seed)
        train_loader, test_loader = create_mnist_loaders(ROOT / "data", config.batch_size)
        print(f"Mode: {'quick validation' if QUICK_RUN else 'full experiment'}")
        print(f"Device: {device}")
        """),
        markdown("""
        ## Train the baseline and association

        Checkpoints, configuration, and loss history are saved under `outputs/`,
        which is intentionally excluded from Git.
        """),
        code("""
        model = Autoencoder(config.capacity, config.latent_dims, config.context_dim)
        history = train_experiment(
            model, train_loader, config, device, output_dir, max_batches=max_batches
        )
        print(f"Total parameters: {count_parameters(model):,}")
        print(f"Trainable after freezing: {count_parameters(model, trainable_only=True):,}")
        print(f"Artifacts: {[path.name for path in sorted(output_dir.iterdir())]}")
        """),
        code("""
        baseline_x = range(1, len(history.baseline_loss) + 1)
        association_x = range(len(history.baseline_loss) + 1,
                              len(history.baseline_loss) + len(history.association_loss) + 1)
        plt.figure(figsize=(7, 4))
        plt.plot(baseline_x, history.baseline_loss, marker="o", label="Baseline")
        plt.plot(association_x, history.association_loss, marker="o", label="Association")
        plt.xlabel("Epoch")
        plt.ylabel("Mean squared error")
        plt.title("Two-stage training history")
        plt.legend()
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(ROOT / "assets" / "training_history.png", dpi=160)
        plt.show()
        """),
        markdown("## Reconstruction examples"),
        code("""
        model.eval()
        images, _ = next(iter(test_loader))
        images = images.to(device)
        noisy = add_noise(images, config.image_noise)
        context = generate_context(
            len(images), noise_level=config.context_noise,
            category_id=config.context_channel, context_dim=config.context_dim, device=device
        )
        with torch.inference_mode():
            reconstructed = model(noisy, context)
        save_reconstruction_grid(
            images, noisy, reconstructed, ROOT / "assets" / "reconstruction_comparison.png"
        )
        """),
        markdown("""
        **Interpretation.** A quick run validates the pipeline, but it is not sufficient
        evidence for a scientific conclusion. Use full training before reporting final
        reconstruction quality or context effects.
        """),
    ],
)


write(
    "03_context_evaluation.ipynb",
    [
        markdown("""
        # 3. Context evaluation

        This notebook loads the association model and compares identical noisy inputs
        under inactive and active context. It reports reconstruction error and the
        average change in the latent representation.

        Run notebook 2 first. Both notebooks must use the same `QUICK_RUN` setting.
        """),
        code(COMMON),
        code("""
        import torch

        from src.config import ExperimentConfig
        from src.data import create_mnist_loaders
        from src.evaluation import reconstruction_mse
        from src.models import Autoencoder
        from src.utils import add_noise, generate_context, seed_everything
        from src.visualization import save_context_comparison

        QUICK_RUN = True
        output_dir = ROOT / "outputs" / ("notebook_quick" if QUICK_RUN else "portfolio")
        required = [output_dir / "config.json", output_dir / "association.pt"]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Run notebook 02 first. Missing: {missing}")

        config = ExperimentConfig.load(output_dir / "config.json")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        seed_everything(config.seed)
        _, test_loader = create_mnist_loaders(ROOT / "data", config.batch_size)
        model = Autoencoder(config.capacity, config.latent_dims, config.context_dim).to(device)
        model.load_state_dict(torch.load(
            output_dir / "association.pt", map_location=device, weights_only=True
        ))
        model.eval()
        print(f"Loaded: {output_dir.relative_to(ROOT) / 'association.pt'}")
        print(f"Device: {device}")
        """),
        markdown("## Quantitative reconstruction comparison"),
        code("""
        max_batches = 20 if QUICK_RUN else None
        mse_without = reconstruction_mse(
            model, test_loader, config, device, use_context=False, max_batches=max_batches
        )
        seed_everything(config.seed)
        mse_with = reconstruction_mse(
            model, test_loader, config, device, use_context=True, max_batches=max_batches
        )
        print(f"MSE without context: {mse_without:.6f}")
        print(f"MSE with context:    {mse_with:.6f}")
        print(f"Context delta:       {mse_with - mse_without:+.6f}")
        """),
        markdown("## Visual and latent comparison"),
        code("""
        images, labels = next(iter(test_loader))
        images = images.to(device)
        noisy = add_noise(images, config.image_noise)
        inactive = generate_context(
            len(images), noise_level=config.context_noise,
            category_id=config.context_channel, context_dim=config.context_dim, device=device
        )
        active = inactive.clone()
        active[:, config.context_channel] = config.context_signal

        with torch.inference_mode():
            latent_inactive = model.encoder(noisy, inactive)
            latent_active = model.encoder(noisy, active)
            reconstruction_inactive = model.decoder(latent_inactive)
            reconstruction_active = model.decoder(latent_active)

        latent_shift = (latent_active - latent_inactive).norm(dim=1).mean().item()
        output_shift = (reconstruction_active - reconstruction_inactive).abs().mean().item()
        print(f"Mean latent L2 shift: {latent_shift:.6f}")
        print(f"Mean absolute pixel shift: {output_shift:.6f}")
        save_context_comparison(
            images, reconstruction_inactive, reconstruction_active,
            ROOT / "assets" / "context_effect.png"
        )
        """),
        markdown("""
        **Interpretation.** The latent and pixel shifts confirm that the context channel
        affects model behavior. The sign and magnitude of the MSE delta must be judged
        after full training and alongside target-specific behavioral analysis; a lower
        global MSE alone does not demonstrate a perceptual bias toward digit `3`.
        """),
    ],
)

print(f"Generated 3 notebooks in {NOTEBOOKS}")
