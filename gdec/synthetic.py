"""Synthetic dataset generation."""
from typing import Tuple

import numpy as np
from scipy import stats

from gdec import npgp, useful


def generate_dataset(
    amplitude=1.0, lengthscale=0.1, n_classes=32, n_data=1024, n_features=32, seed=1434
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic dataset for testing decoders.

    Args:
        amplitude: The Fourier-domain GP prior amplitude for the weight matrix, in
            [0, infinity).
        lengthscale: The Fourier domain GP prior lengthscale for the weight matrix, in
            [0, 1].
        n_classes: The number of classes in the dataset.
        n_data: The number of examples in the datset.
        n_features: The number of features/neurons in the dataset.
        seed: The random seed. The default, 1434, is chosen to generate a dataset with
            a low discrepancy between the number of examples in the least common class
            and the number of examples in the most common class.

    Returns:
        A tuple (X, y) of examples, of shape (n_data, n_features) and labels, of shape
        (n_data, ). The labels are integers in {0, ..., n_classes - 1}.

    """
    np.random.seed(seed)
    x_mean = np.zeros((n_features,))
    x_cov = stats.invwishart(df=n_features, scale=0.1 * np.eye(n_features)).rvs()
    x_dist = stats.multivariate_normal(mean=x_mean, cov=x_cov)
    X = x_dist.rvs(n_data)

    def spectrum(w: np.ndarray) -> np.ndarray:
        """GP power spectrum, evaluated elementwise."""
        return npgp.rbf_spectrum(w, amplitude, lengthscale)

    n_funs = npgp.choose_n_basis_funs(spectrum)
    basis = npgp.whitened_fourier_basis(spectrum, n_domain=n_classes, n_funs=n_funs)
    spectral_coefs = np.random.randn(n_funs, n_features)
    W = basis @ spectral_coefs

    probs = np.exp(useful.log_softmax(X @ W.T, axis=-1))
    classes = useful.categorical_sample(probs)
    return X, classes
