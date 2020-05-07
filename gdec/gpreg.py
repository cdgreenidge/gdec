"""1D Gaussian Process regression, accelerated with Fourier methods."""
import math
import warnings

import jax
import jax.numpy as np
import numpy as onp
import sklearn.base
from jax import config
from jax.scipy import linalg, special
from scipy import optimize
from sklearn.utils import validation
from torch import quasirandom

from gdec import jaxgp


def theta_pushforward(theta_unconstrained: np.ndarray) -> np.ndarray:
    """Push unconstrained hyperparameters forward to constrained values."""
    noise = np.exp(theta_unconstrained[0])
    amplitude = np.exp(theta_unconstrained[1])
    lengthscale = 0.001 + 0.998 * special.expit(theta_unconstrained[2])
    return np.array([noise, amplitude, lengthscale])


def theta_pullback(theta_constrained: np.ndarray) -> np.ndarray:
    """Pull back batches constrained hyperparams to unconstrained values."""
    noise = np.log(theta_constrained[0])
    amplitude = np.log(theta_constrained[1])
    lengthscale = special.logit((theta_constrained[2] - 0.001) / 0.998)
    return np.stack((noise, amplitude, lengthscale), axis=0)


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
    noise, amplitude, lengthscale = theta_pushforward(theta)
    sigma2 = noise ** 2
    raw_spectrum = jaxgp.rbf_spectrum(spectrum_freqs, amplitude, lengthscale)
    spectrum = np.where(raw_spectrum, 0, raw_spectrum > 1.0e-32)
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
        phi.T @ phi + (noise ** 2) * np.eye(phi.shape[1]), phi.T @ y, sym_pos=True
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
        self,
        n_classes,
        noise_initial=1.0,
        amplitude_initial=1.0,
        lengthscale_initial=0.5,
    ):
        if config.values["jax_enable_x64"] == 0:
            warnings.warn(
                (
                    "PeriodicGPRegression: "
                    "JAX running in 32 bit mode, NaN/Inf errors likely"
                ),
                UserWarning,
            )
        self.noise_initial = noise_initial
        self.amplitude_initial = amplitude_initial
        self.lengthscale_initial = lengthscale_initial
        self.loss = jax.jit(neg_log_evidence, static_argnums=(3, 4))
        self.vloss = jax.jit(
            jax.vmap(self.loss, in_axes=(0, None, None, None, None), out_axes=0)
        )
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
        basis, spectrum_freqs = jaxgp.fourier_basis(self.grid_size_, self.grid_size_)

        # Do an initial grid search to find a good initialization
        sobol = quasirandom.SobolEngine(dimension=3, scramble=True, seed=43)
        draws = sobol.draw(1024).numpy()

        noise_lower = 0.1 * np.std(y)
        noise_upper = 0.5 * np.std(y)
        noises = (noise_upper - noise_lower) * draws[:, 0] + noise_lower

        amplitude_lower = 0.1
        amplitude_upper = np.max(np.abs(y))
        if amplitude_upper < amplitude_lower:
            amplitude_upper = 2.0
        amplitudes = (amplitude_upper - amplitude_lower) * draws[:, 1] + amplitude_lower

        lengthscale_lower = 0.001
        log_lengthscale_lower = math.log(lengthscale_lower)
        lengthscale_upper = 0.95
        log_lengthscale_upper = math.log(lengthscale_upper)
        log_lengthscales = (log_lengthscale_upper - log_lengthscale_lower) * draws[
            :, 2
        ] + log_lengthscale_lower
        lengthscales = np.exp(log_lengthscales)

        theta_0s = theta_pullback(
            np.stack((noises, amplitudes, lengthscales), axis=0)
        ).T

        losses = self.vloss(theta_0s, x, y, basis, spectrum_freqs)

        self.theta_0_ = theta_0s[onp.argmin(losses)]

        # Fit hyperparameters
        def minimize_loss(
            noise_initial, amplitude_initial, lengthscale_initial
        ) -> optimize.OptimizeResult:
            args = (x, y, basis, spectrum_freqs)
            return optimize.minimize(
                lambda theta: onp.asarray(self.loss(theta, *args)),
                self.theta_0_,
                method="trust-ncg",
                jac=lambda theta: onp.asarray(self.loss_grad(theta, *args)),
                hess=lambda theta: onp.asarray(self.loss_hess(theta, *args)),
            )

        results = minimize_loss(
            self.noise_initial, self.amplitude_initial, self.lengthscale_initial
        )
        theta_est = results.x
        self.noise_, self.amplitude_, self.lengthscale_ = theta_pushforward(theta_est)

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
        validation.check_is_fitted(self, "is_fitted_")
        return onp.asarray(self.f_[np.squeeze(X)])
