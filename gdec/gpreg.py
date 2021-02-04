"""1D Gaussian Process regression, accelerated with Fourier methods."""
from typing import Optional, Tuple

import numpy as np
import sklearn.base
from scipy import linalg, optimize
from sklearn.utils import validation

from gdec import npgp, useful


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


def param_pushforward(theta: np.ndarray) -> np.ndarray:
    """Convert an array of regular hyperparameters to optimization-domain params.

    Args:
        theta: An array of shape (3, ) or (b, 3), where b is a batch dimension, holding
            the noise standard deviation (sigma, not sigma^2), kernel amplitude
            (p, not p^2), and kernel lengthscale (l, not l^2).

    Returns:
        An array of shape (3, ) or (b, 3) containing the noise standard deviation,
        log rho tilde (i.e. log(rho^2 * l)), and log lengthscale^2, i.e. log(l^2).

    """
    if theta.ndim == 1:
        theta = np.expand_dims(theta, 0)  # Add batch dimension
    assert theta.ndim == 2
    noises = theta[:, 0]
    amps = theta[:, 1]
    lens = theta[:, 2]
    cs, ds = npgp.rbf_natural_to_unconstrained(amps, lens)
    out = np.array([noises, cs, ds]).T
    return np.squeeze(out)


def param_pullback(theta: np.ndarray) -> np.ndarray:
    """Convert an array of optimizaion-domain hyperparams to regular params.

    Args:
        theta: An array of shape (3, ) or (b, 3), where b is a batch dimension
            containing the noise standard deviation, log rho tilde
            (i.e. log(rho^2 * l)), and log lengthscale^2, i.e. log(l^2).

    Returns:
        theta: An array of shape (3, ) or (b, 3), where b is a batch dimension, holding
            the noise standard deviation (sigma, not sigma^2), kernel amplitude
            (p, not p^2), and kernel lengthscale (l, not l^2).

    """
    if theta.ndim == 1:
        theta = np.expand_dims(theta, 0)  # Add batch dimension
    assert theta.ndim == 2
    noises = theta[:, 0]
    log_rho_tildes = theta[:, 1]
    log_len2s = theta[:, 2]
    lens = np.sqrt(np.exp(log_len2s))
    out = np.array([noises, np.sqrt(np.exp(log_rho_tildes) / lens), lens]).T
    return np.squeeze(out)


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
    """Prunes the spectrum and its associated matrices."""
    condition_thresh = 1e6
    spectrum[np.isnan(spectrum)] = 0
    spectrum_thresh = np.max(np.abs(spectrum)) / condition_thresh
    (mask,) = np.nonzero(
        np.logical_and(np.abs(spectrum) > spectrum_thresh, np.abs(spectrum) > 1e-32)
    )
    spectrum_freqs = spectrum_freqs[mask]
    spectrum = spectrum[mask]
    phixphix = phixphix[mask, :][:, mask]
    phixy = phixy[mask]
    return spectrum_freqs, spectrum, phixphix, phixy


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
    spectrum = npgp.rbf_spectrum(spectrum_freqs, amplitude, lengthscale)

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
    """Compute the gradient of the negative log likelihood.

    Args:
        theta: An array of shape (3, ) containing the unconstrained hyperparameters
            (noise, amplitude, and lengthscale)
        spectrum_freqs: The Fourier frequencies associated with each function in the
            basis, of shape (n_funs, ).

    Returns:
        The gradient of theta, of shape (3, ).

    """
    sigma, amplitude, lengthscale = theta
    sigma2 = sigma ** 2
    sigma3 = sigma ** 3
    spectrum = npgp.rbf_spectrum(spectrum_freqs, amplitude, lengthscale)

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
    drhoK_diag = npgp.rbf_spectrum_dc(spectrum_freqs, amplitude, lengthscale)
    drhoA_diag = -((Kinv_diag) ** 2) * drhoK_diag
    dlK_diag = npgp.rbf_spectrum_dd(spectrum_freqs, amplitude, lengthscale)
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
    """Compute the Hessian of the negative log likelihood.

    This code could probably be condensed significantly but it does produce
    correct results.

    Args:
        theta: An array of shape (3, ) containing the unconstrained hyperparameters
            (noise, amplitude, and lengthscale)
        spectrum_freqs: The Fourier frequencies associated with each function in the
            basis, of shape (n_funs, ).

    Returns:
        The Hessian of theta, of shape (3, 3).

    """
    sigma, amplitude, lengthscale = theta
    spectrum = npgp.rbf_spectrum(spectrum_freqs, amplitude, lengthscale)
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

    drhoK = np.diag(npgp.rbf_spectrum_dc(spectrum_freqs, amplitude, lengthscale))
    dlK = np.diag(npgp.rbf_spectrum_dd(spectrum_freqs, amplitude, lengthscale))
    drhodrhoK = np.diag(npgp.rbf_spectrum_dcdc(spectrum_freqs, amplitude, lengthscale))
    dldlK = np.diag(npgp.rbf_spectrum_dddd(spectrum_freqs, amplitude, lengthscale))
    dldrK = np.diag(npgp.rbf_spectrum_dcdd(spectrum_freqs, amplitude, lengthscale))

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
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute a MAP estimate of the posterior.

    Args:
        X: An array of shape (n_samples, 1). The training input samples. Must
            lie on the grid {0, ..., k - 1}.
        y: An array of shape (n_samples, ). The training target values.
        whitened_basis: The whitened Fourier basis, of shape (k, n_funs).
        noise: The estimated observation noise standard deviation.

    Returns:
        The estimated posterior mean for the grid {0, ..., k - 1}, a vector of shape
        (k, ) and the diagonal posterior variance.

    """
    phi = whitened_basis[x]
    w = linalg.solve(  # Estimated spectral-domain weights
        phi.T @ phi + (noise ** 2) * np.eye(phi.shape[1]), phi.T @ y, sym_pos=True
    )
    cov = linalg.inv((1 / noise ** 2) * phi.T @ phi + np.eye(phi.shape[1]))  # w domain
    cov_diag = np.diag(whitened_basis @ cov @ whitened_basis.T)
    return whitened_basis @ w, cov_diag  # Estimated latent function values


class PeriodicGPRegression(sklearn.base.BaseEstimator):
    """1D Gaussian Process regression with circular boundary conditions.

    We use an RBF kernel (currently this is fixed).

    Args:
        noise_initial: The intitial value of the output noise standard deviation.
        amplitude_initial: The initial value of the kernel amplitude.
        lengthscale_initial: The initial value of the kernel lengthscale.

    """

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

        basis, self.spectrum_freqs_ = npgp.real_fourier_basis(self.grid_size_)
        self.sstats_ = sufficient_statistics(x, y, basis)

        # Do an initial grid search to find a good initialization
        # The below code is not as clean as it could be, but it matches Jonathan's
        # implementation exactly
        n_points = 4
        yy = self.sstats_[3]
        n_samples = y.size
        y_var = yy / n_samples  # We assume that y is centered

        log_noise_var_lower = np.log(min(1, y_var * 0.05))
        log_noise_var_upper = np.log(y_var)
        log_noise_range = _grid_range(
            log_noise_var_lower, log_noise_var_upper, n_points
        )
        noises = np.sqrt(np.exp(np.linspace(*log_noise_range, n_points)))

        log_amplitude2_lower = np.log(min(1, y_var * 0.05))
        log_amplitude2_upper = np.log(y_var)

        log_lengthscale_min = np.log(0.1)
        log_lengthscale_max = np.log(np.ptp(X) / 2)
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

        theta_0s = useful.product(noises, cs, ds)

        losses = []
        for i in theta_0s:
            losses.append(nll(i, self.sstats_, self.spectrum_freqs_))

        self.theta_0_ = theta_0s[np.argmin(losses)]

        # Fit hyperparameters
        def minimize_loss() -> optimize.OptimizeResult:
            args = (self.sstats_, self.spectrum_freqs_)
            return optimize.minimize(
                lambda theta: nll(theta, *args),
                self.theta_0_,
                # trust-exact is faster but is brittle and sometimes fails on
                # poorly-conditioned problems
                method="trust-ncg",
                jac=lambda theta: nll_grad(theta, *args),
                hess=lambda theta: nll_hess(theta, *args),
                # Setting max_trust_radius prevents overflow of the log-domain
                # parameters
                options={"max_trust_radius": 32.0},
            )

        results = minimize_loss()
        self.noise_, self.c_, self.d_ = results.x
        self.amplitude_, self.lengthscale_ = npgp.rbf_unconstrained_to_natural(
            self.c_, self.d_
        )

        # Fit latent function with MAP estimate
        spectrum = npgp.rbf_spectrum(self.spectrum_freqs_, self.c_, self.d_)
        whitened_basis = np.sqrt(spectrum)[None, :] * basis
        self.f_, self.f_var_ = map_estimate(x, y, whitened_basis, self.noise_)

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

    def nll(self, theta: Optional[np.ndarray] = None) -> float:
        """Evaluate the negative log likelihood.

        Args:
            theta: An array of shape (3, ) containing the noise standard
            deviation, the amplitude standard deviation, and the lengthscale. If
            None, the trained hyperparameters are used.

        """
        validation.check_is_fitted(self, "is_fitted_")
        if theta is None:
            theta = np.array([self.noise_, self.amplitude_, self.lengthscale_])
        return nll(param_pushforward(theta), self.sstats_, self.spectrum_freqs_)
