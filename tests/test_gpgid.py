"""Test gpgid.py."""
from typing import Tuple

import numpy as np
import pytest
from sklearn import model_selection

import gdec
from gdec import synthetic


@pytest.fixture(scope="module")
def dataset() -> Tuple[np.ndarray, np.ndarray]:
    return synthetic.generate_dataset(
        seed=1634, examples_per_class=8, n_classes=32, n_features=2
    )


def test_you_can_train_gppid_on_the_synthetic_dataset(dataset):
    X, y = dataset
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, train_size=0.8
    )
    model = gdec.GPGaussianIndependentDecoder()
    model.fit(X_train, y_train, verbose=False)
    score = model.score(X_test, y_test)
    assert score > 1 / 32  # Better than random guessing?
