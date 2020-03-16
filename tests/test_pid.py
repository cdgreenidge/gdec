"""Test pid.py."""
from typing import Tuple

import numpy as np
import pytest

from gdec import pid, synthetic


@pytest.fixture(scope="module")
def dataset() -> Tuple[np.ndarray, np.ndarray]:
    return synthetic.generate_dataset(seed=1434, n_classes=32)


def test_training_pid_on_float_dataset_raises_value_error(dataset):
    X, y = dataset
    X = X.astype(np.float64)
    model = pid.PID()
    with pytest.raises(ValueError):
        model.fit(X, y)
