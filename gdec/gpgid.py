"""Gaussian-process regularized Gaussian independent decoder."""
import logging
from typing import List, Tuple

import numpy as np
import sklearn.naive_bayes
import tqdm
from scipy import stats

from gdec import gpreg

logger = logging.getLogger(__name__)


def tuning_curve_matrix(
    X: np.ndarray, y: np.ndarray, verbose: bool
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calcualtes a smooth tuning curve matrix.

    Args:
        X: The array of spike counts, of shape (n, d).
        y: The array of class labels, of shape (n, ).
        verbose: Whether or not to print a progress bar.

    Returns:
        The smoothed tuning curve matrix, of shape `(k, d)` where `k` is the number of
        classes, and the estimated amplitudes, lengthscales, and observation noise
        parameters, of shape `(d, )`.

    """
    curves: List[np.ndarray] = []
    amplitudes: List[float] = []
    lengthscales: List[float] = []
    noises: List[float] = []
    grid = np.arange(np.unique(y).size)
    logger.info("Smoothing tuning curves...")
    for i in tqdm.tqdm(range(X.shape[1]), disable=not verbose):
        model = gpreg.PeriodicGPRegression().fit(y[:, None], X[:, i])
        curves.append(model.predict(grid[:, None]))
        amplitudes.append(model.amplitude_)
        lengthscales.append(model.lengthscale_)
        noises.append(model.noise_)

    return (
        np.stack(curves, axis=1),
        np.array(amplitudes),
        np.array(lengthscales),
        np.array(noises),
    )


class GPGaussianIndependentDecoder(sklearn.naive_bayes._BaseDiscreteNB):
    """Poisson Naive Bayes classifier."""

    def _joint_log_likelihood(self, X):
        """Compute the unnormalized posterior log probability of X.

        Args:
            X: An array of shape ``(n_samples, n_features)`` containing the examples.

        Returns
            An array of shape ``(n_samples, n_classes)``, containing ``log P(c) +
            log P(x|c)`` for all rows of X.

        """
        sklearn.utils.validation.check_is_fitted(self)
        X = sklearn.utils.check_array(X)

        joint_log_likelihood = np.zeros((len(self.classes_), X.shape[0]))
        for i in range(len(self.classes_)):
            mean = self.coefs_[i]
            log_prior = np.log(self.class_prior_[i])
            joint_log_likelihood[i, :] = (
                stats.norm.logpdf(X, mean, self.noises_).sum(axis=1) + log_prior
            )

        return joint_log_likelihood.T

    def fit(
        self, X: np.ndarray, y: np.ndarray, verbose: bool = True
    ) -> "GPGaussianIndependentDecoder":
        """Fit Poisson Naive Bayes according to X, y.

        Args:
            X: An array of shape ``(n_samples, n_features)`` containing the training
                examples.
            y: An array of shape ``(n_samples, )`` containing the training labels.
            verbose: Whether or not to print a progress bar.

        Returns:
            self, an object.
            mean = self.coefs_[i]
            log_prior = np.log(self.class_prior_[i])
            joint_log_likelihood[i, :] = (
                stats.norm.logpdf(X, mean, self.noises_).sum(axis=1) + log_prior
            )

        """
        X, y = sklearn.utils.check_X_y(X, y)

        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        self.class_count_ = np.zeros(n_classes)
        for i, y_i in enumerate(self.classes_):
            self.class_count_[i] = np.sum(y == y_i)

        out = tuning_curve_matrix(X, y, verbose)
        self.coefs_, self.amplitudes_, self.lengthscales_, self.noises_ = out

        self.class_prior_ = self.class_count_ / self.class_count_.sum()
        return self
