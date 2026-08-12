import pytest
import torch

from src.models import Autoencoder, count_parameters, freeze_except_context


def test_autoencoder_preserves_image_shape():
    model = Autoencoder(capacity=4, latent_dims=8, context_dim=3)
    images = torch.randn(2, 1, 28, 28)
    context = torch.randn(2, 3)
    assert model(images, context).shape == images.shape


def test_encoder_rejects_invalid_context_shape():
    model = Autoencoder(context_dim=3)
    with pytest.raises(ValueError, match="context must have shape"):
        model(torch.randn(2, 1, 28, 28), torch.randn(2, 4))


def test_freeze_except_context_only_leaves_projection_trainable():
    model = Autoencoder(capacity=4, latent_dims=8, context_dim=3)
    total = count_parameters(model)
    freeze_except_context(model)
    trainable_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    assert trainable_names == {"encoder.fc_context.weight", "encoder.fc_context.bias"}
    assert 0 < count_parameters(model, trainable_only=True) < total
