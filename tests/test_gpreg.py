"""Test gpreg.py."""
from typing import Tuple

import numpy as np
import pytest

from gdec import gpreg, npgp


@pytest.fixture(scope="module")
def dataset() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    np.random.seed(42)
    amplitude = 1.0
    lengthscale = 12
    sigma = 0.5
    n = 128

    def spectrum_fn(w: np.ndarray) -> np.ndarray:
        return npgp.rbf_spectrum(w, amplitude, lengthscale)

    basis, freqs = npgp.real_fourier_basis(n)
    coef_vars = npgp.rbf_spectrum(freqs, amplitude, lengthscale)
    z = np.arange(n)

    w = np.sqrt(coef_vars) * np.random.randn(n)
    f = basis @ w

    x = z.repeat(1)
    f_x = f.repeat(1)
    y = sigma * np.random.randn(*f_x.shape) + f_x
    return x[:, None], y, z[:, None], f


def test_you_can_train_periodic_gp_regression_on_the_synthetic_dataset(dataset):
    X, y, z, f = dataset
    model = gpreg.PeriodicGPRegression(n_classes=np.unique(X).size)
    model.fit(X, y)
    f_est = model.predict(z)
    error = np.max(np.abs(f - f_est))
    assert error < 0.3


def test_training_pid_on_float_dataset_raises_value_error(dataset):
    X, y, _, _ = dataset
    X = X.astype(np.float32)
    model = gpreg.PeriodicGPRegression(n_classes=np.unique(X).size)
    with pytest.raises(ValueError):
        model.fit(X, y)
