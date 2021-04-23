"""Gaussian-process regularized Poisson Independent Decoder."""
import logging
from typing import List, Tuple

import numpy as np
import sklearn.naive_bayes
import tqdm
from scipy import optimize, stats

from gdec import npgp, useful

logger = logging.getLogger(__name__)


def _grid_range(low: float, hi: float, ngrid: int) -> Tuple[float, float]:
    """Calculate a grid range according to Jonathan's code.

    For some reason his code doesn't evaluate on the grid endpoints.

    Args:
        low: The low end of the range.
        hi: The high end of the range.

    Returns:
        The range along which to calculate the grid points.

    """
    len_ = hi - low
    offset = len_ / (2 * ngrid)
    return low + offset, hi - offset


def loss(
    coefs: np.ndarray, x: np.ndarray, y: np.ndarray, basis: np.ndarray
) -> np.ndarray:
    """Negative log likelihood loss for the GP latent model with exp-Pois likelihood.

    Args:
        coefs: The spectral cofficients for the GP, of shape `(d, )`
        x: The integer-valued x-locations of the data, of shape `(n, )`.
        y: The spike counts (responses), of shape `(n, )`.
        basis: The whitened Fourier basis used to represent the latent function.

    Returns:
        The negative log likelihood loss of the coefficients.

    """
    log_f = basis[x] @ coefs
    log_likelihood = np.sum(y * log_f - np.exp(log_f))
    log_prior = -0.5 * np.dot(coefs, coefs)
    return -(log_likelihood + log_prior)


def loss_grad(
    coefs: np.ndarray, x: np.ndarray, y: np.ndarray, basis: np.ndarray
) -> np.ndarray:
    """Calculate the gradient of the negative log likelihood loss.

    Args:
        coefs: The spectral cofficients for the GP, of shape `(d, )`
        x: The integer-valued x-locations of the data, of shape `(n, )`.
        y: The spike counts (responses), of shape `(n, )`.
        basis: The whitened Fourier basis used to represent the latent function.

    Returns:
        The gradient of the negative log likelihood loss of the coefficients, of shape
        `(d, )`.

    """
    phi = basis[x]  # Expand basis for each element in input
    t1 = phi.T @ (y - np.exp(phi @ coefs))
    t2 = -coefs
    return -(t1 + t2)


def loss_hess(
    coefs: np.ndarray, x: np.ndarray, y: np.ndarray, basis: np.ndarray
) -> np.ndarray:
    """Calculate the Hessian of the negative log likelihood loss.

    Args:
        coefs: The spectral cofficients for the GP, of shape `(d, )`
        x: The integer-valued x-locations of the data, of shape `(n, )`.
        y: The spike counts (responses), of shape `(n, )`.
        basis: The whitened Fourier basis used to represent the latent function.

    Returns:
        The Hessian of the negative log likelihood loss of the coefficients, of shape
        `(d, d)`.

    """
    phi = basis[x]  # Expand basis for each element in input
    t1 = -np.identity(coefs.size)
    t2 = -(phi.T * np.exp(phi @ coefs)[None, :]) @ phi
    return -(t1 + t2)


def fit_latent(
    x: np.ndarray, y: np.ndarray, c: np.ndarray, d: np.ndarray
) -> Tuple[optimize.OptimizeResult, np.ndarray]:
    """Fit the latent function for the GP model with exp-Pois likelihood.

    Args:
        x: The integer-valued x-locations of the data, of shape `(n, )`.
        y: The spike counts (responses), of shape `(n, )`.
        c: The first kernel parameter.
        lengthscale: The second kernel parameter.

    Returns:
        The SciPy optimization result object for the coefficients, and the whitened
        Fourier basis that was constructed.

    """
    n_classes = np.unique(x).size
    basis, spectrum_freqs = npgp.real_fourier_basis(n_classes)
    spectrum = npgp.rbf_spectrum(spectrum_freqs, c, d)
    basis = basis * np.sqrt(spectrum[None, :])  # Whiten the basis

    condition_thresh = 1e8
    spectrum_thresh = np.max(np.abs(spectrum)) / condition_thresh
    (mask,) = np.nonzero(np.abs(spectrum) > spectrum_thresh)
    n_funs = mask.size
    basis = basis[:, mask]

    initial_coefs = np.zeros((n_funs,))
    results = optimize.minimize(
        loss,
        initial_coefs,
        args=(x, y, basis),
        method="trust-ncg",
        jac=loss_grad,
        hess=loss_hess,
    )
    return (results, basis)


def neg_log_laplace_evidence(
    unconstrained_hyperparams: np.ndarray, x: np.ndarray, y: np.ndarray
) -> np.ndarray:
    """Calculate the negative log evidence using the Laplace approximation.

    Args:
        unconstrained_hyperparams: An array with two elements. The first is the log
            prior amplitude, the second is the arctanh prior lengthscale.
        x: The integer-valued x-locations of the data, of shape `(n, )`.
        y: The spike counts (responses), of shape `(n, )`.

    Returns:
        The approximate negative log evidence.

    """
    results = fit_latent(
        x, y, unconstrained_hyperparams[0], unconstrained_hyperparams[1]
    )[0]
    return -(-results.fun - 0.5 * np.linalg.slogdet(results.hess)[1])


def fit_hyperparams(
    x: np.ndarray,
    y: np.ndarray,
    initial_amplitude: float = 10.0,
    initial_lengthscale: float = 0.01,
) -> Tuple[float, float]:
    """Estimate optimal hyperparameters using the Laplace approximation.

    Args:
        x: The integer-valued x-locations of the data, of shape `(n, )`.
        y: The spike counts (responses), of shape `(n, )`.
        initial_amplitude: The initial guess for the prior amplitude.
        initial_lengthscale: The intial guess for the prior lengthscale.

    Returns:
        The amplitude and lengthscale estimated to maximize the log evidence.

    """
    # Do an initial grid search
    n_points = 4
    yy = y @ y
    n_samples = y.size
    y_var = yy / n_samples  # We assume that y is centered

    log_amplitude2_lower = np.log(min(1, y_var * 0.05))
    log_amplitude2_upper = np.log(y_var)

    log_lengthscale_min = np.log(0.1)
    log_lengthscale_max = np.log(np.ptp(x) / 2)
    log_lengthscale_range = _grid_range(
        log_lengthscale_min, log_lengthscale_max, n_points
    )
    lengthscales = np.exp(np.linspace(*log_lengthscale_range, n_points))

    trho_min = log_amplitude2_lower + np.log(
        np.sqrt(2 * np.pi) * np.exp(log_lengthscale_min)
    )
    trho_max = log_amplitude2_upper + np.log(
        np.sqrt(2 * np.pi) * np.exp(log_lengthscale_max)
    )
    trho_range = _grid_range(trho_min, trho_max, n_points)
    trhos = np.linspace(*trho_range, n_points)
    rhos = np.sqrt(np.exp(trhos) / lengthscales)

    cs, ds = npgp.rbf_natural_to_unconstrained(rhos, lengthscales)

    theta_0s = useful.product(cs, ds)

    losses = []
    for i in theta_0s:
        losses.append(neg_log_laplace_evidence(i, x, y))

    theta_0_ = theta_0s[np.argmin(losses)]

    results = optimize.minimize(
        neg_log_laplace_evidence, theta_0_, args=(x, y), method="Nelder-Mead",
    )
    return results.x[0], results.x[1]


def smooth_curve(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """Smooth a tuning curve using a latent GP firing rate with exp-Pois likelihood.

    Args:
        x: The integer-valued x-locations of the data, of shape `(n, )`.
        y: The spike counts (responses), of shape `(n, )`.

    Returns:
        The smoothed tuning curve for each value in x (sorted), and the estimated
        amplitude and lengthscale of the latent prior.

    """
    est_c, est_d = fit_hyperparams(x, y)
    est_amplitude, est_lengthscale = npgp.rbf_unconstrained_to_natural(est_c, est_d)
    results, basis = fit_latent(x, y, est_c, est_d)
    log_f_curve_est = basis @ results.x
    return log_f_curve_est, est_amplitude, est_lengthscale


def tuning_curve_matrix(
    X: np.ndarray, y: np.ndarray, verbose: bool
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calcualtes a smooth tuning curve matrix.

    Args:
        X: The array of spike counts, of shape (n, d).
        y: The array of class labels, of shape (n, ).
        verbose: Whether or not to print a progress bar.

    Returns:
        The smoothed tuning curve matrix, of shape `(k, d)` where `k` is the number of
        classes, and the estimated amplitudes and lengthscales, of shape `(d, )`.

    """
    curves: List[np.ndarray] = []
    amplitudes: List[float] = []
    lengthscales: List[float] = []

    logger.info("Smoothing tuning curves...")
    for i in tqdm.tqdm(range(X.shape[1]), disable=not verbose):
        log_curve, amplitude, lengthscale = smooth_curve(y, X[:, i])
        curves.append(log_curve)
        amplitudes.append(amplitude)
        lengthscales.append(lengthscale)

    return np.stack(curves, axis=1), np.array(amplitudes), np.array(lengthscales)


class GPPoissonIndependentDecoder(sklearn.naive_bayes._BaseDiscreteNB):
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
            mean = np.exp(self.coefs_[i])
            log_prior = np.log(self.class_prior_[i])
            joint_log_likelihood[i, :] = (
                stats.poisson.logpmf(X, mean).sum(axis=1) + log_prior
            )

        return joint_log_likelihood.T

    def fit(
        self, X: np.ndarray, y: np.ndarray, verbose: bool = True
    ) -> "GPPoissonIndependentDecoder":
        """Fit Poisson Naive Bayes according to X, y.

        Args:
            X: An array of shape ``(n_samples, n_features)`` containing the training
                examples.
            y: An array of shape ``(n_samples, )`` containing the training labels.
            verbose: Whether or not to print a progress bar.

        Returns:
            self, an object.

        """
        X, y = sklearn.utils.check_X_y(X, y)

        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        self.class_count_ = np.zeros(n_classes)
        for i, y_i in enumerate(self.classes_):
            self.class_count_[i] = np.sum(y == y_i)

        self.coefs_, self.amplitudes_, self.lengthscales_ = tuning_curve_matrix(
            X, y, verbose
        )

        self.class_prior_ = self.class_count_ / self.class_count_.sum()
        return self
