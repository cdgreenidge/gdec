"""Test glmnet.py."""
import warnings
from typing import Tuple

import numpy as np
import pytest
from sklearn import exceptions, model_selection

import gdec
from gdec import synthetic


@pytest.fixture(scope="module")
def dataset() -> Tuple[np.ndarray, np.ndarray]:
    return synthetic.generate_dataset(seed=1634, n_classes=4)


def test_you_can_train_glmnet_on_the_synthetic_dataset(dataset):
    X, y = dataset
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, train_size=0.8
    )
    model = gdec.LogisticRegression()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", exceptions.ConvergenceWarning)
        model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    assert score > 1 / 4  # Better than random guessing?


def test_you_can_train_glmnet_with_an_intercept(dataset):
    X, y = dataset
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, train_size=0.8
    )
    model = gdec.LogisticRegression(affine=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", exceptions.ConvergenceWarning)
        model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    assert score > 1 / 4  # Better than random guessing?
