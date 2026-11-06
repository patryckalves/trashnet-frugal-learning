"""Testes unitários para as arquiteturas baseline MLP e CNN."""

import numpy as np
import pytest

from src.models import build_cnn, build_mlp


def test_build_mlp_architecture() -> None:
    """Verifica construção e dimensionalidade de saída do classificador MLP."""
    pytest.importorskip("tensorflow")

    mlp_model = build_mlp(
        input_shape=(96, 128, 3),
        num_classes=6,
        hidden_units=(128, 64),
        dropout_rate=0.2,
    )
    dummy_input = np.random.randn(2, 96, 128, 3).astype(np.float32)
    output = mlp_model.predict(dummy_input, verbose=0)

    assert output.shape == (2, 6)
    assert np.allclose(output.sum(axis=1), 1.0, atol=1e-5)


def test_build_cnn_shape_invariance() -> None:
    """Verifica invariância dimensional da CNN através do GlobalAveragePooling2D.

    Comprova que uma mesma rede com GlobalAveragePooling2D processa resoluções espaciais
    distintas (128x96, 256x192, 512x384) mantendo os mesmos pesos parametrizados.
    """
    pytest.importorskip("tensorflow")

    cnn_model = build_cnn(
        input_shape=(None, None, 3),
        num_classes=6,
        conv_filters=(16, 32),
        dense_units=32,
    )

    for h, w in [(96, 128), (192, 256), (384, 512)]:
        sample_tensor = np.random.randn(1, h, w, 3).astype(np.float32)
        out = cnn_model.predict(sample_tensor, verbose=0)
        assert out.shape == (1, 6)
        assert np.allclose(out.sum(axis=1), 1.0, atol=1e-5)
