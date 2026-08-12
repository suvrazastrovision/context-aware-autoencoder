import pytest
import torch

from src.utils import add_noise, generate_context, seed_everything


def test_context_signal_activates_selected_channel():
    seed_everything(7)
    context = generate_context(5, signal_level=2.0, category_id=1, context_dim=3)
    assert context.shape == (5, 3)
    assert torch.all(context[:, 1] == 2.0)


def test_noise_preserves_shape_and_range():
    images = torch.zeros(3, 1, 28, 28)
    noisy = add_noise(images, noise_level=3.0, correlated=True)
    assert noisy.shape == images.shape
    assert noisy.min() >= -1
    assert noisy.max() <= 1


def test_context_rejects_invalid_channel():
    with pytest.raises(ValueError, match="valid context channel"):
        generate_context(2, category_id=3, context_dim=3)
