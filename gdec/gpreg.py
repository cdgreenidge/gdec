"""1D Gaussian Process regression, accelerated with Fourier methods."""
import math

import jax
import jax.numpy as np
import numpy as onp
import sklearn.base
from jax.scipy import linalg, special
from scipy import optimize
from sklearn.utils import validation

from gdec import jaxgp


def neg_log_evidence(
    theta: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    basis: np.ndarray,
    spectrum_freqs: np.ndarray,
) -> np.ndarray:
    """Compute the negative log evidence given some hyperparameters.

    Args:
        theta: An array of shape (3, ) containing the unconstrained hyperparameters
            (noise, amplitude, and lengthscale)
        X: An array of shape (n_samples, 1). The training input samples. Must
            lie on the grid {0, ..., k - 1}.
        y: An array of shape (n_samples, ). The training target values.
        basis: The unwhitened Fourier basis, of shape (k, n_funs).
        spectrum_freqs: The Fourier frequencies associated with each function in the
            basis, of shape (n_funs, ).

    Returns:
        A scalar, the negative log evidence.

    """
    sigma2 = np.exp(theta[0]) ** 2
    amplitude = np.exp(theta[1])
    lengthscale = 0.95 * special.expit(theta[2]) + 0.025
    spectrum = jaxgp.rbf_spectrum(spectrum_freqs, amplitude, lengthscale)
    whitened_basis = np.sqrt(spectrum)[None, :] * basis
    phi = whitened_basis[x]
    beta = 1 / sigma2
    A = beta * phi.T @ phi + np.eye(phi.shape[1])
    b = beta * y.T @ phi
    n = x.shape[0]

    t1 = n * np.log(sigma2)
    t2 = beta * y.T @ y

    t3 = np.linalg.slogdet(A)[1]
    t4 = -b.T @ linalg.solve(A, b, sym_pos=True)

    return 0.5 * (t1 + t2 + t3 + t4)


@jax.jit
def map_estimate(
    x: np.ndarray, y: np.ndarray, whitened_basis: np.ndarray, noise: float
) -> np.ndarray:
    """Compute a MAP estimate of the posterior.

    Args:
        X: An array of shape (n_samples, 1). The training input samples. Must
            lie on the grid {0, ..., k - 1}.
        y: An array of shape (n_samples, ). The training target values.
        whitened_basis: The whitened Fourier basis, of shape (k, n_funs).
        noise: The estimated observation noise standard deviation.

    Returns:
        The estimated posterior mean for the grid {0, ..., k - 1}, a vector of shape
        (k, ).

    """
    phi = whitened_basis[x]
    w = linalg.solve(  # Estimated spectral-domain weights
        phi.T @ phi + (noise ** 2) * np.eye(phi.shape[1]), phi.T @ y, sym_pos=True,
    )
    return whitened_basis @ w  # Estimated latent function values


class PeriodicGPRegression(sklearn.base.BaseEstimator):
    """1D Gaussian Process regression with circular boundary conditions.

    We use an RBF kernel (currently this is fixed).

    Args:
        noise_initial: The intitial value of the output noise standard deviation.
        amplitude_initial: The initial value of the kernel amplitude.
        lengthscale_initial: The initial value of the kernel lengthscale.

    """

    def __init__(
        self, noise_initial=0.25, amplitude_initial=1.0, lengthscale_initial=0.25,
    ):
        self.noise_initial = noise_initial
        self.amplitude_initial = amplitude_initial
        self.lengthscale_initial = lengthscale_initial
        self.n_funs = jaxgp.choose_n_basis_funs(
            lambda w: jaxgp.rbf_spectrum(
                w, self.amplitude_initial, self.lengthscale_initial
            )
        )
        self.loss = jax.jit(neg_log_evidence, static_argnums=(3, 4))
        self.loss_grad = jax.jit(jax.grad(self.loss))
        self.loss_hess = jax.jit(jax.hessian(self.loss))

    def fit(self, X, y, grid_size: int = None):
        """Fit the Gaussian process regression.

        Args:
            X: An array of shape (n_samples, 1). The training input samples. Must
                lie on the grid {0, ..., k - 1}.
            y: An array of shape (n_samples, ). The target values.
            grid_max: The size of the grid (i.e., k).

        Returns:
            Self.

        """
        X, y = validation.check_X_y(X, y)
        x = np.squeeze(X)
        if x.dtype.kind not in ("i", "u"):
            raise ValueError("X must be an array with int/unit dtype.")
        self.grid_size_ = onp.unique(x).size if grid_size is None else grid_size

        # It's cleaner to use the whitened_fourier_basis in the loss function but that
        # triggers JAX recompilation on every evaulation, leading to slow refits
        basis, spectrum_freqs = jaxgp.fourier_basis(self.grid_size_, self.n_funs)

        # Fit hyperparameters
        unconstrained_lengthscale = special.logit(np.array((0.1 - 0.025) / 0.95)).item()
        theta_0 = np.array(
            [
                math.log(self.noise_initial),
                math.log(self.amplitude_initial),
                unconstrained_lengthscale,
            ]
        )
        args = (x, y, basis, spectrum_freqs)
        results = optimize.minimize(
            lambda theta: onp.asarray(self.loss(theta, *args)),
            theta_0,
            method="trust-exact",
            jac=lambda theta: onp.asarray(self.loss_grad(theta, *args)),
            hess=lambda theta: onp.asarray(self.loss_hess(theta, *args)),
        )
        theta_est = results.x
        self.noise_ = np.exp(theta_est[0])
        self.amplitude_ = np.exp(theta_est[1])
        self.lengthscale_ = 0.95 * special.expit(theta_est[2]) + 0.025

        # Fit latent function with MAP estimate
        spectrum = jaxgp.rbf_spectrum(
            spectrum_freqs, self.amplitude_, self.lengthscale_
        )
        whitened_basis = np.sqrt(spectrum)[None, :] * basis
        self.f_ = map_estimate(x, y, whitened_basis, self.noise_)

        self.is_fitted_ = True
        return self

    def predict(self, X):
        """Predict the function values.

        Args:
            X : array of shape (n_samples, ). The test input samples. Must lie on the
                same grid as the train samples.

        Returns:
            y: array of shape (n_samples, ). The predictions for each value of X.

        """
        X = validation.check_array(X, accept_sparse=True)
        if X.dtype.kind not in ("i", "u"):
            raise ValueError("X must be an array with int/unit dtype.")
        validation.check_is_fitted(self, "is_fitted_")
        return onp.asarray(self.f_[np.squeeze(X)])
