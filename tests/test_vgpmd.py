"""Test vgpmd.py."""
from typing import Tuple

import numpy as np
import pytest
from sklearn import model_selection

import gdec
from gdec import synthetic


@pytest.fixture(scope="module")
def dataset() -> Tuple[np.ndarray, np.ndarray]:
    return synthetic.generate_dataset(
        seed=1634, examples_per_class=8, n_classes=8, n_features=8
    )


def test_you_can_train_the_gpmd_on_the_synthetic_dataset(dataset):
    X, y = dataset
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, train_size=0.8
    )
    model = gdec.GaussianProcessMulticlassDecoder()
    model.fit(X_train, y_train, max_steps=128)
    score = model.score(X_test, y_test,)
    assert score > 1 / 32  # Better than random guessing?
    assert model.amplitudes_.size == X.shape[1]
    assert model.lengthscales_.size == X.shape[1]


def test_gpmd_can_train_with_intercept(dataset):
    X, y = dataset
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, train_size=0.8,
    )
    model = gdec.GaussianProcessMulticlassDecoder(affine=True)
    model.fit(X_train, y_train, max_steps=128)
