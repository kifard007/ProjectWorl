import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from smap_lstm.data import make_sequences


def test_make_sequences_shapes_and_values():
    values = np.arange(20, dtype=np.float32).reshape(10, 2)
    X, y = make_sequences(values, window_size=3)
    assert X.shape == (7, 3, 2)
    assert y.shape == (7, 2)
    np.testing.assert_array_equal(X[0], values[:3])
    np.testing.assert_array_equal(y, values[3:])
    assert np.shares_memory(X, values)

