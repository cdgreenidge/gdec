"""1D Gaussian Process regression, accelerated with Fourier methods."""
import math
from typing import Tuple

import numpy as np
import sklearn.base
from scipy import linalg, optimize
from sklearn.utils import validation
from torch import quasirandom

from gdec import jaxgp


def sufficient_statistics(
    x: np.ndarray, y: np.ndarray, basis: np.ndarray
) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """Compute the sufficient statistics for model fitting.

    Args:
        X: An array of shape (n_samples, ). The training input samples. Must
            lie on the grid {0, ..., k - 1}.
        y: An array of shape (n_samples, ). The training target values.
        basis: The unwhitened Fourier basis, of shape (k, n_funs).

    Returns:
        A tuple of the sufficient statistics.

    """
    return x.shape[0], basis[x].T @ basis[x], basis[x].T @ y, y.T @ y


def prune(
    spectrum_freqs: np.ndarray,
    spectrum: np.ndarray,
    phixphix: np.ndarray,
    phixy: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Prunes the spectrum and its matrices."""
    condition_thresh = 1e8
    spectrum_thresh = np.max(np.abs(spectrum)) / condition_thresh
    (mask,) = np.nonzero(np.abs(spectrum) > spectrum_thresh)
    spectrum_freqs = spectrum_freqs[mask]
    spectrum = spectrum[mask]
    phixphix = phixphix[mask, :][:, mask]
    phixy = phixy[mask]
    return spectrum_freqs, spectrum, phixphix, phixy


def rbf_spectrum(
    w: np.ndarray, amplitude: np.ndarray, lengthscale: np.ndarray
) -> np.ndarray:
    """Evaluate the Matern 5/2 power spectrum element-wise at ``w``.

    Args:
        w: The (dimensionless) frequencies at which to evaluate the power spectrum, of
            shape (n, ).
        amplitude: The kernel amplitude, can be batched with shape (b, ).
        lengthscale: The kernel lengthscale, can be batched with shape (b, ).

    Returns:
        The Matern 5/2 spectrum evaluated at U, of shape (b, n).

    """
    return (
        amplitude ** 2
        * np.sqrt(2 * math.pi * lengthscale ** 2)
        * np.exp(-2 * math.pi ** 2 * lengthscale ** 2 * w ** 2)
    )


def rbf_spectrum_dr(
    w: np.ndarray, amplitude: np.ndarray, lengthscale: np.ndarray
) -> np.ndarray:
    """Evaluate the RBF spectrum derivative w.r.t. rho.

    Args:
        w: The (dimensionless) frequencies at which to evaluate the power spectrum, of
            shape (n, ).
        amplitude: The kernel amplitude, can be batched with shape (b, ).
        lengthscale: The kernel lengthscale, can be batched with shape (b, ).

    Returns:
        The RBF spectrum derivative at U, of shape (b, n).

    """
    return (
        2
        * amplitude
        * np.sqrt(2 * math.pi * lengthscale ** 2)
        * np.exp(-2 * math.pi ** 2 * lengthscale ** 2 * w ** 2)
    )


def rbf_spectrum_dl(
    w: np.ndarray, amplitude: np.ndarray, lengthscale: np.ndarray
) -> np.ndarray:
    """Evaluate the RBF spectrum derivative w.r.t. rho.

    Args:
        w: The (dimensionless) frequencies at which to evaluate the power spectrum, of
            shape (n, ).
        amplitude: The kernel amplitude, can be batched with shape (b, ).
        lengthscale: The kernel lengthscale, can be batched with shape (b, ).

    Returns:
        The RBF spectrum derivative at U, of shape (b, n).

    """
    return (
        amplitude ** 2
        * np.sqrt(2 * math.pi)
        * np.exp(-2 * math.pi ** 2 * lengthscale ** 2 * w ** 2)
        * (1 - 4 * math.pi ** 2 * lengthscale ** 2 * w ** 2)
    )


def rbf_spectrum_drdr(
    w: np.ndarray, amplitude: np.ndarray, lengthscale: np.ndarray
) -> np.ndarray:
    """Evaluate the RBF spectrum derivative w.r.t. rho.

    Args:
        w: The (dimensionless) frequencies at which to evaluate the power spectrum, of
            shape (n, ).
        amplitude: The kernel amplitude, can be batched with shape (b, ).
        lengthscale: The kernel lengthscale, can be batched with shape (b, ).

    Returns:
        The RBF spectrum derivative at U, of shape (b, n).

    """
    return (
        2
        * np.sqrt(2 * math.pi * lengthscale ** 2)
        * np.exp(-2 * math.pi ** 2 * lengthscale ** 2 * w ** 2)
    )


def rbf_spectrum_dldl(
    w: np.ndarray, amplitude: np.ndarray, lengthscale: np.ndarray
) -> np.ndarray:
    """Evaluate the RBF spectrum derivative w.r.t. rho.

    Args:
        w: The (dimensionless) frequencies at which to evaluate the power spectrum, of
            shape (n, ).
        amplitude: The kernel amplitude, can be batched with shape (b, ).
        lengthscale: The kernel lengthscale, can be batched with shape (b, ).

    Returns:
        The RBF spectrum derivative at U, of shape (b, n).

    """
    return (
        -(amplitude ** 2)
        * np.sqrt(2 * math.pi)
        * np.exp(-2 * math.pi ** 2 * lengthscale ** 2 * w ** 2)
        * 4
        * math.pi ** 2
        * lengthscale
        * w ** 2
        * (3 - 4 * math.pi ** 2 * lengthscale ** 2 * w ** 2)
    )


def rbf_spectrum_dldr(
    w: np.ndarray, amplitude: np.ndarray, lengthscale: np.ndarray
) -> np.ndarray:
    """Evaluate the RBF spectrum derivative w.r.t. rho.

    Args:
        w: The (dimensionless) frequencies at which to evaluate the power spectrum, of
            shape (n, ).
        amplitude: The kernel amplitude, can be batched with shape (b, ).
        lengthscale: The kernel lengthscale, can be batched with shape (b, ).

    Returns:
        The RBF spectrum derivative at U, of shape (b, n).

    """
    return (
        2
        * amplitude
        * np.sqrt(2 * math.pi)
        * np.exp(-2 * math.pi ** 2 * lengthscale ** 2 * w ** 2)
        * (1 - 4 * math.pi ** 2 * lengthscale ** 2 * w ** 2)
    )


def nll(
    theta: np.ndarray,
    sstats: Tuple[int, np.ndarray, np.ndarray, np.ndarray],
    spectrum_freqs: np.ndarray,
) -> np.ndarray:
    """Compute the negative log evidence given some hyperparameters.

    Args:
        theta: An array of shape (3, ) containing the unconstrained hyperparameters
            (noise, amplitude, and lengthscale)
        spectrum_freqs: The Fourier frequencies associated with each function in the
            basis, of shape (n_funs, ).

    Returns:
        A scalar, the negative log evidence.

    """
    sigma, amplitude, lengthscale = theta
    sigma2 = sigma ** 2
    spectrum = rbf_spectrum(spectrum_freqs, amplitude, lengthscale)

    # Compute sufficient statistics
    n, phixphix, phixy, yy = sstats
    spectrum_freqs, spectrum, phixphix, phixy = prune(
        spectrum_freqs, spectrum, phixphix, phixy
    )

    beta = 1 / sigma2
    A = beta * phixphix + np.diag(1 / spectrum)
    b = beta * phixy
    (L, lower) = linalg.cho_factor(A)

    return 0.5 * (
        n * np.log(2 * np.pi)
        + n * np.log(sigma2)
        + beta * yy
        + np.log(spectrum).sum()
        + 2 * np.sum(np.log(np.diag(L)))
        - b.T @ linalg.cho_solve((L, lower), b)
    )


def nll_grad(
    theta: np.ndarray, sstats: np.ndarray, spectrum_freqs: np.ndarray
) -> np.ndarray:
    sigma, amplitude, lengthscale = theta
    sigma2 = sigma ** 2
    sigma3 = sigma ** 3
    spectrum = rbf_spectrum(spectrum_freqs, amplitude, lengthscale)

    n, phixphix, phixy, yy = sstats
    spectrum_freqs, spectrum, phixphix, phixy = prune(
        spectrum_freqs, spectrum, phixphix, phixy
    )

    beta = 1 / sigma2
    A = beta * phixphix + np.diag(1 / spectrum)
    b = beta * phixy
    Ainv = linalg.inv(A)
    Ainvb = Ainv @ b
    dsigmaA = -(2 / sigma3) * phixphix
    dsigmab = -(2 / sigma3) * phixy

    Kinv_diag = 1 / spectrum
    drhoK_diag = rbf_spectrum_dr(spectrum_freqs, amplitude, lengthscale)
    drhoA_diag = -((Kinv_diag) ** 2) * drhoK_diag
    dlK_diag = rbf_spectrum_dl(spectrum_freqs, amplitude, lengthscale)
    dlA_diag = -((Kinv_diag) ** 2) * dlK_diag

    dsigma = (
        n / sigma
        - yy / sigma3
        + 0.5 * (Ainv * dsigmaA).sum()
        - Ainvb @ (dsigmab - 0.5 * dsigmaA @ Ainvb)
    )

    drho = 0.5 * (
        (np.diag(Ainv) * drhoA_diag + Kinv_diag * drhoK_diag).sum()
        + np.dot(Ainvb * drhoA_diag, Ainvb)
    )

    dl = 0.5 * (
        (np.diag(Ainv) * dlA_diag + Kinv_diag * dlK_diag).sum()
        + np.dot(Ainvb * dlA_diag, Ainvb)
    )

    return np.array([dsigma, drho, dl])


def nll_hess(
    theta: np.ndarray, sstats: np.ndarray, spectrum_freqs: np.ndarray
) -> np.ndarray:
    sigma, amplitude, lengthscale = theta
    spectrum = rbf_spectrum(spectrum_freqs, amplitude, lengthscale)
    n, phixphix, phixy, yy = sstats
    spectrum_freqs, spectrum, phixphix, phixy = prune(
        spectrum_freqs, spectrum, phixphix, phixy
    )

    A = (1 / sigma ** 2) * phixphix + np.diag(1 / spectrum)
    b = (1 / sigma ** 2) * phixy
    Ainv = linalg.inv(A)
    Ainvb = Ainv @ b

    Kinv_diag = 1 / spectrum
    Kinv = np.diag(Kinv_diag)
    Kinv2 = np.diag(Kinv_diag ** 2)
    Kinv3 = np.diag(Kinv_diag ** 3)

    dsigmaA = -(2 / sigma ** 3) * phixphix
    dsigma2A = (6 / sigma ** 4) * phixphix
    dsigmab = -(2 / sigma ** 3) * phixy
    dsigma2b = (6 / sigma ** 4) * phixy
    dsigmaAinvb = -Ainv @ dsigmaA @ Ainvb + Ainv @ dsigmab

    drhoK = np.diag(rbf_spectrum_dr(spectrum_freqs, amplitude, lengthscale))
    dlK = np.diag(rbf_spectrum_dl(spectrum_freqs, amplitude, lengthscale))
    drhodrhoK = np.diag(rbf_spectrum_drdr(spectrum_freqs, amplitude, lengthscale))
    dldlK = np.diag(rbf_spectrum_dldl(spectrum_freqs, amplitude, lengthscale))
    dldrK = np.diag(rbf_spectrum_dldr(spectrum_freqs, amplitude, lengthscale))

    drhoA = np.diag(drhoK @ -((Kinv_diag) ** 2))
    dlA = np.diag(dlK @ -((Kinv_diag) ** 2))

    dsigmadsigma = (
        -(n / sigma ** 2)
        + 3 * yy / sigma ** 4
        - 0.5 * np.sum((Ainv @ dsigmaA) ** 2)
        + 0.5 * np.trace(Ainv @ dsigma2A)
        + Ainv @ dsigmab @ (dsigmaA @ Ainvb - dsigmab)
        - Ainvb.T @ dsigma2b
        + Ainv @ (dsigmab - dsigmaA @ Ainvb) @ dsigmaA @ Ainvb
        + 0.5 * (Ainvb.T @ dsigma2A @ Ainvb)
    )

    def dt1dt2(dt1K: np.ndarray, dt2K: np.ndarray, dt1t2K: np.ndarray) -> np.ndarray:
        """Compute the Hessian entry w.r.t. kernel params t1 and t2."""
        dt2A = -np.diag(Kinv_diag ** 2) @ dt2K
        return (
            0.5
            * np.trace(
                Kinv
                @ (dt1t2K - dt2K @ Kinv @ dt1K)
                @ (np.eye(Ainv.shape[0]) - Kinv @ Ainv)
                + dt1K @ Kinv @ (dt2K @ Kinv + Ainv @ dt2A) @ Ainv @ Kinv
            )
            + Ainvb.T @ dt2A @ Ainv @ dt1K @ Kinv2 @ Ainvb
            - 0.5
            * Ainvb.T
            @ ((dt1t2K - dt1K @ dt2K @ Kinv) @ Kinv2 - dt1K @ dt2K @ Kinv3)
            @ Ainvb
        )

    def dsigmadt(dtA: np.ndarray) -> np.ndarray:
        """Compute the Hessian entry w.r.t. sigma and kernal param t."""
        return (
            -0.5 * np.trace(dtA @ Ainv @ dsigmaA @ Ainv) + dsigmaAinvb.T @ dtA @ Ainvb
        )

    drhodrho = dt1dt2(drhoK, drhoK, drhodrhoK)
    dldl = dt1dt2(dlK, dlK, dldlK)
    dldr = dt1dt2(dlK, drhoK, dldrK)
    dsigmadrho = dsigmadt(drhoA)
    dsigmadl = dsigmadt(dlA)

    return np.array(
        [
            [dsigmadsigma, dsigmadrho, dsigmadl],
            [dsigmadrho, drhodrho, dldr],
            [dsigmadl, dldr, dldl],
        ]
    )


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
        self.noise_initial = noise_initial
        self.amplitude_initial = amplitude_initial
        self.lengthscale_initial = lengthscale_initial

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
        self.grid_size_ = np.unique(x).size if grid_size is None else grid_size

        # It's cleaner to use the whitened_fourier_basis in the loss function but that
        # triggers JAX recompilation on every evaulation, leading to slow refits
        basis, spectrum_freqs = jaxgp.fourier_basis(self.grid_size_, self.grid_size_)
        sstats = sufficient_statistics(x, y, basis)

        # Do an initial grid search to find a good initialization
        sobol = quasirandom.SobolEngine(dimension=3, scramble=True, seed=43)
        draws = sobol.draw(16).numpy()

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

        theta_0s = np.stack((noises, amplitudes, lengthscales), axis=0).T

        losses = []
        for i in theta_0s:
            losses.append(nll(i, sstats, spectrum_freqs))

        self.theta_0_ = theta_0s[np.argmin(losses)]

        # Fit hyperparameters
        def minimize_loss(
            noise_initial, amplitude_initial, lengthscale_initial
        ) -> optimize.OptimizeResult:
            args = (sstats, spectrum_freqs)
            return optimize.minimize(
                lambda theta: nll(theta, *args),
                self.theta_0_,
                method="trust-exact",
                jac=lambda theta: nll_grad(theta, *args),
                hess=lambda theta: nll_hess(theta, *args),
            )

        results = minimize_loss(
            self.noise_initial, self.amplitude_initial, self.lengthscale_initial
        )
        self.noise_, self.amplitude_, self.lengthscale_ = results.x
        self.lengthscale_ = np.abs(self.lengthscale_)

        # Fit latent function with MAP estimate
        spectrum = rbf_spectrum(spectrum_freqs, self.amplitude_, self.lengthscale_)
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
        return self.f_[np.squeeze(X)]
