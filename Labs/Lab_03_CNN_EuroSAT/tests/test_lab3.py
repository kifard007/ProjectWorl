import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eurosat_cnn.data import split_dataset, validate_images


def _fake_data(n_per_class=25, n_classes=10):
    labels = np.repeat(np.arange(n_classes), n_per_class)
    images = np.random.default_rng(0).integers(0, 256, (len(labels), 64, 64, 3), dtype=np.uint8)
    return images, labels


def test_split_sizes_and_stratification():
    images, labels = _fake_data()
    split = split_dataset(images, labels, test_fraction=0.2, validation_fraction=0.2)
    # 250 снимков: 20% в test, затем 20% от оставшихся 200 в validation.
    assert len(split.X_test) == 50
    assert len(split.X_val) == 40
    assert len(split.X_train) == 160
    # Каждый класс представлен во всех выборках поровну.
    for part in (split.y_train, split.y_val, split.y_test):
        counts = np.bincount(part, minlength=10)
        assert counts.min() == counts.max()


def test_split_has_no_overlap():
    images, labels = _fake_data()
    # Делаем каждое изображение уникальным, чтобы проверить пересечения.
    images[:, 0, 0, 0] = np.arange(len(images)) % 256
    images[:, 0, 0, 1] = np.arange(len(images)) // 256
    split = split_dataset(images, labels)
    keys = [set(map(bytes, part[:, 0, 0, :2])) for part in (split.X_train, split.X_val, split.X_test)]
    assert not keys[0] & keys[1]
    assert not keys[0] & keys[2]
    assert not keys[1] & keys[2]
    assert sum(map(len, keys)) == len(images)


def test_validate_rejects_wrong_shape():
    with pytest.raises(ValueError):
        validate_images(np.zeros((2, 32, 32, 3), dtype=np.uint8), np.array([0, 1]))


def test_model_structure_and_output():
    pytest.importorskip("tensorflow")
    from eurosat_cnn.model import build_cnn_model

    model = build_cnn_model()
    layer_types = {type(layer).__name__ for layer in model.layers}
    assert {"Conv2D", "MaxPooling2D", "Dropout", "Dense"} <= layer_types
    assert model.get_layer("augmentation") is not None

    batch = np.zeros((4, 64, 64, 3), dtype=np.uint8)
    probabilities = model.predict(batch, verbose=0)
    assert probabilities.shape == (4, 10)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, rtol=1e-5)