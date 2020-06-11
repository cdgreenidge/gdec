"""1D Gaussian Process regression, accelerated with Fourier methods."""
from typing import Optional, Tuple

import numpy as np
import sklearn.base
from scipy import linalg, optimize
from sklearn.utils import validation

from gdec import npgp, useful


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
    spectrum_thresh = np.max(np.abs(spectrum)) / condition_thresh
    (mask,) = np.nonzero(np.abs(spectrum) > spectrum_thresh)
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
    n, phixphix, phixy, yy = sstats
    spectrum_freqs, spectrum, phixphix, phixy = prune(
        spectrum_freqs, spectrum, phixphix, phixy
    )

    beta = 1 / sigma2
    Kinv_diag = 1 / spectrum
    phixphix_diag = np.diag(phixphix)
    A_diag = beta * phixphix_diag + Kinv_diag
    b = beta * phixy
    Ainvb = b / A_diag

    return 0.5 * (
        n * np.log(2 * np.pi)
        + n * np.log(sigma2)
        + beta * yy
        + np.log(spectrum).sum()
        + np.log(A_diag).sum()
        - b @ Ainvb
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
    spectrum = npgp.rbf_spectrum(spectrum_freqs, amplitude, lengthscale)
    n, phixphix, phixy, yy = sstats
    spectrum_freqs, spectrum, phixphix, phixy = prune(
        spectrum_freqs, spectrum, phixphix, phixy
    )

    Kinv_diag = 1 / spectrum
    Kinv2_diag = Kinv_diag ** 2
    phixphix_diag = np.diag(phixphix)
    A_diag = (1 / sigma ** 2) * phixphix_diag + Kinv_diag
    b = (1 / sigma ** 2) * phixy
    Ainv_diag = 1 / A_diag
    Ainvb = Ainv_diag * b

    drhoK_diag = npgp.rbf_spectrum_dr(spectrum_freqs, amplitude, lengthscale)
    drhoA_diag = -Kinv2_diag * drhoK_diag
    dlK_diag = npgp.rbf_spectrum_dl(spectrum_freqs, amplitude, lengthscale)
    dlA_diag = -Kinv2_diag * dlK_diag
    dsigmaA_diag = -(2 / sigma ** 3) * phixphix_diag
    dsigmab = -(2 / sigma ** 3) * phixy

    dsigma = (
        n / sigma
        - yy / (sigma ** 3)
        + 0.5 * Ainv_diag @ dsigmaA_diag
        - Ainvb @ (dsigmab - 0.5 * dsigmaA_diag * Ainvb)
    )

    drho = 0.5 * (
        Ainv_diag @ drhoA_diag
        + Kinv_diag @ drhoK_diag
        + np.dot(Ainvb * drhoA_diag, Ainvb)
    )

    dl = 0.5 * (
        Ainv_diag @ dlA_diag + Kinv_diag @ dlK_diag + np.dot(Ainvb * dlA_diag, Ainvb)
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
    # Note: basis is orthogonal, so phixhpix is diagonal
    (n, phixphix, phixy, yy,) = sstats
    spectrum_freqs, spectrum, phixphix, phixy = prune(
        spectrum_freqs, spectrum, phixphix, phixy
    )

    Kinv_diag = 1 / spectrum
    Kinv2_diag = Kinv_diag ** 2
    phixphix_diag = np.diag(phixphix)
    A_diag = (1 / sigma ** 2) * phixphix_diag + Kinv_diag
    b = (1 / sigma ** 2) * phixy
    Ainv_diag = 1 / A_diag
    Ainvb = Ainv_diag * b

    dsigmaA_diag = -(2 / sigma ** 3) * phixphix_diag
    dsigma2A_diag = (6 / sigma ** 4) * phixphix_diag
    dsigmab = -(2 / sigma ** 3) * phixy
    dsigma2b = (6 / sigma ** 4) * phixy
    dsigmaAinvb = -Ainv_diag * (dsigmaA_diag * Ainvb - dsigmab)

    drhoK_diag = npgp.rbf_spectrum_dr(spectrum_freqs, amplitude, lengthscale)
    dlK_diag = npgp.rbf_spectrum_dl(spectrum_freqs, amplitude, lengthscale)
    drhodrhoK_diag = npgp.rbf_spectrum_drdr(spectrum_freqs, amplitude, lengthscale)
    dldlK_diag = npgp.rbf_spectrum_dldl(spectrum_freqs, amplitude, lengthscale)
    dldrK_diag = npgp.rbf_spectrum_dldr(spectrum_freqs, amplitude, lengthscale)

    drhoA_diag = -drhoK_diag * Kinv2_diag
    dlA_diag = -dlK_diag * Kinv2_diag

    dsigmadsigma = (
        -(n / sigma ** 2)
        + 3 * yy / (sigma ** 4)
        - 0.5 * np.sum((Ainv_diag * dsigmaA_diag) ** 2)
        + 0.5 * Ainv_diag @ dsigma2A_diag
        + dsigmaAinvb @ (Ainvb * dsigmaA_diag - dsigmab)
        + 0.5 * Ainvb @ (Ainvb * dsigma2A_diag - 2 * dsigma2b)
    )

    def dt1dt2(
        dt1K_diag: np.ndarray, dt2K_diag: np.ndarray, dt1t2K_diag: np.ndarray
    ) -> np.ndarray:
        """Compute the Hessian entry w.r.t. kernel params t1 and t2."""
        C = Kinv_diag - Ainv_diag * Kinv2_diag
        D = dt1K_diag * dt2K_diag * C
        E = dt1t2K_diag - D
        return 0.5 * (np.sum(C * E) + Ainvb @ ((Kinv2_diag * (D - E)) * Ainvb))

    def dsigmadt(dtA_diag: np.ndarray) -> np.ndarray:
        """Compute the Hessian entry w.r.t. sigma and kernal param t."""
        return -0.5 * (
            np.sum(dtA_diag * dsigmaA_diag * Ainv_diag ** 2)
            + dsigmaAinvb @ (dtA_diag * Ainvb)
        )

    drhodrho = dt1dt2(drhoK_diag, drhoK_diag, drhodrhoK_diag)
    dldl = dt1dt2(dlK_diag, dlK_diag, dldlK_diag)
    dldr = dt1dt2(dlK_diag, drhoK_diag, dldrK_diag)
    dsigmadrho = dsigmadt(drhoA_diag)
    dsigmadl = dsigmadt(dlA_diag)

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
        basis, self.spectrum_freqs_ = npgp.real_fourier_basis(self.grid_size_)
        self.sstats_ = sufficient_statistics(x, y, basis)

        # Do an initial grid search to find a good initialization
        n_points = 8
        yy = self.sstats_[3]
        n_samples = y.size
        y_var = yy / n_samples  # We assume that y is centered

        log_noise_lower = np.log(min(1, y_var * 0.05))
        log_noise_upper = np.log(y_var)
        noises = np.exp(np.linspace(log_noise_lower, log_noise_upper, 4))

        log_amplitude_lower = np.log(min(1, y_var * 0.05))
        log_amplitude_upper = np.log(y_var)
        amplitudes = np.exp(
            np.linspace(log_amplitude_lower, log_amplitude_upper, n_points)
        )

        log_lengthscale_min = np.log(0.1)
        log_lengthscale_max = np.log(np.ptp(X) / 2)
        lengthscales = np.exp(
            np.linspace(log_lengthscale_min, log_lengthscale_max, n_points)
        )
        theta_0s = useful.product(noises, amplitudes, lengthscales)
        # import pdb; pdb.set_trace()

        losses = []
        for i in theta_0s:
            losses.append(nll(i, self.sstats_, self.spectrum_freqs_))

        self.theta_0_ = theta_0s[np.argmin(losses)]

        # Fit hyperparameters
        def minimize_loss(
            noise_initial, amplitude_initial, lengthscale_initial
        ) -> optimize.OptimizeResult:
            args = (self.sstats_, self.spectrum_freqs_)
            return optimize.minimize(
                lambda theta: nll(theta, *args),
                self.theta_0_,
                method="trust-exact",
                jac=lambda theta: nll_grad(theta, *args),
                hess=lambda theta: nll_hess(theta, *args),
                options={"gtol": 1.0e-34},
            )

        results = minimize_loss(
            self.noise_initial, self.amplitude_initial, self.lengthscale_initial
        )
        self.noise_, self.amplitude_, self.lengthscale_ = results.x
        self.lengthscale_ = np.abs(self.lengthscale_)

        # Fit latent function with MAP estimate
        spectrum = npgp.rbf_spectrum(
            self.spectrum_freqs_, self.amplitude_, self.lengthscale_
        )
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

    def nll(self, hyperparams: Optional[np.ndarray] = None) -> float:
        """Evaluate the negative log likelihood.

        Args:
            hyperparams: An array of shape (3, ) containing the noise standard
            deviation, the amplitude standard deviation, and the lengthscale. If
            None, the trained hyperparameters are used.

        """
        validation.check_is_fitted(self, "is_fitted_")
        if hyperparams is None:
            hyperparams = np.array([self.noise_, self.amplitude_, self.lengthscale_])
        return nll(hyperparams, self.sstats_, self.spectrum_freqs_)
