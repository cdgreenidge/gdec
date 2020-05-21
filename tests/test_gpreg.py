"""Test gpreg.py."""
from typing import Tuple

import jax.numpy as np
import pytest
from jax import random

from gdec import gpreg, jaxgp


@pytest.fixture(scope="module")
def dataset() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    amplitude = 1.0
    lengthscale = 0.2
    sigma = 0.5

    def spectrum_fn(w: np.ndarray) -> np.ndarray:
        return jaxgp.rbf_spectrum(w, amplitude, lengthscale)

    n_funs = jaxgp.choose_n_basis_funs(spectrum_fn, 24)
    basis = jaxgp.whitened_fourier_basis(spectrum_fn, 128, n_funs)
    z = np.arange(128)

    key = random.PRNGKey(47)
    key, subkey = random.split(key)
    w = random.normal(subkey, shape=(n_funs,))
    f = basis @ w

    x = z.repeat(1)
    f_x = f.repeat(1)
    key, subkey = random.split(key)
    y = sigma * random.normal(subkey, shape=x.shape) + f_x
    return x[:, None], y, z[:, None], f


def test_you_can_train_periodic_gp_regression_on_the_synthetic_dataset(dataset):
    X, y, z, f = dataset
    model = gpreg.PeriodicGPRegression()
    model.fit(X, y)
    f_est = model.predict(z)
    error = np.max(np.abs(f - f_est))
    assert error < 0.35


def test_training_pid_on_float_dataset_raises_value_error(dataset):
    X, y, _, _ = dataset
    X = X.astype(np.float32)
    model = gpreg.PeriodicGPRegression()
    with pytest.raises(ValueError):
        model.fit(X, y)
