"""Test pid.py."""
from typing import Tuple

import numpy as np
import pytest
from sklearn import model_selection

import gdec
from gdec import synthetic


@pytest.fixture(scope="module")
def dataset() -> Tuple[np.ndarray, np.ndarray]:
    return synthetic.generate_dataset(seed=1634, n_classes=32)


def test_you_can_train_pid_on_the_synthetic_dataset():
    X, y = synthetic.generate_dataset(seed=1634, n_classes=32)
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, train_size=0.8
    )
    model = gdec.PID()
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    assert score > 1 / 32  # Better than random guessing?


def test_training_pid_on_float_dataset_raises_value_error(dataset):
    X, y = dataset
    X = X.astype(np.float64)
    model = gdec.PID()
    with pytest.raises(ValueError):
        model.fit(X, y)
