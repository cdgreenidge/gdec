"""Gaussian process utilities for Numpy code."""
import math
from typing import Tuple

import numpy as np


def real_fourier_basis(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Construct a real unitary fourier basis of size (n).

    Args:
        n: The basis size:

    Returns:
        A tuple of the basis, and the frequencies at which to evaluate
        the spectral distribution function to get the variances for the
        Fourier-domain coefficients.

    """
    assert n > 1
    dc = np.ones((n,))
    dc_freq = 0

    cosine_basis_vectors = []
    cosine_freqs = []
    sine_basis_vectors = []
    sine_freqs = []

    ts = np.arange(n)
    for w in range(1, 1 + (n - 1) // 2):
        x = w * (2 * np.pi / n) * ts
        cosine_basis_vectors.append(np.sqrt(2) * np.cos(x))
        cosine_freqs.append(w)
        sine_basis_vectors.append(-np.sqrt(2) * np.sin(x))
        sine_freqs.append(w)

    if n % 2 == 0:
        w = n // 2
        x = w * 2 * np.pi * ts / n
        cosine_basis_vectors.append(np.cos(x))
        cosine_freqs.append(w)

    basis = np.column_stack((dc, *cosine_basis_vectors, *sine_basis_vectors[::-1]))
    freqs = np.concatenate([[dc_freq], cosine_freqs, sine_freqs[::-1]])

    return basis / np.sqrt(n), freqs / n


def rbf_spectrum(
    w: np.ndarray, log_rho_tilde: np.ndarray, log_lengthscale: np.ndarray
) -> np.ndarray:
    """Evaluate the RBF spectrum.

    Args:
        w: The (dimensionless) frequencies at which to evaluate the power spectrum, of
            shape (n, ).
        log_rho_tilde: log(amplitude * lengthscale ** 2), can be batched with shape
            (b, ).
        log_lengthscale: The kernel log lengthscale, can be batched with shape (b, ).

    Returns:
        The RBF spectrum derivative at U, of shape (b, n).

    """
    return (
        np.exp(log_rho_tilde)
        * np.sqrt(2 * math.pi)
        * np.exp(-2 * math.pi ** 2 * np.exp(log_lengthscale) * w ** 2)
    )


def rbf_spectrum_dr(
    w: np.ndarray, log_rho_tilde: np.ndarray, log_lengthscale: np.ndarray
) -> np.ndarray:
    """Evaluate the RBF spectrum derivative w.r.t log_rho_tilde.

    Args:
        w: The (dimensionless) frequencies at which to evaluate the power spectrum, of
            shape (n, ).
        log_rho_tilde: log(amplitude * lengthscale ** 2), can be batched with shape
            (b, ).
        log_lengthscale: The kernel log lengthscale, can be batched with shape (b, ).

    Returns:
        The RBF spectrum derivative at U, of shape (b, n).

    """
    return (
        np.exp(log_rho_tilde)
        * np.sqrt(2 * math.pi)
        * np.exp(-2 * math.pi ** 2 * np.exp(log_lengthscale) * w ** 2)
    )


def rbf_spectrum_dl(
    w: np.ndarray, log_rho_tilde: np.ndarray, log_lengthscale: np.ndarray
) -> np.ndarray:
    """Evaluate the RBF spectrum derivative w.r.t. log_lengthscale.

    Args:
        w: The (dimensionless) frequencies at which to evaluate the power spectrum, of
            shape (n, ).
        log_rho_tilde: log(amplitude * lengthscale ** 2), can be batched with shape
            (b, ).
        log_lengthscale: The kernel log lengthscale, can be batched with shape (b, ).

    Returns:
        The RBF spectrum derivative at U, of shape (b, n).

    """
    return (
        np.exp(log_rho_tilde)
        * np.sqrt(2 * math.pi)
        * np.exp(-2 * math.pi ** 2 * np.exp(log_lengthscale) * w ** 2)
        * -2
        * math.pi ** 2
        * np.exp(log_lengthscale)
        * w ** 2
    )


def rbf_spectrum_drdr(
    w: np.ndarray, log_rho_tilde: np.ndarray, log_lengthscale: np.ndarray
) -> np.ndarray:
    """Evaluate the RBF spectrum 2nd derivative w.r.t. log_rho_tilde.

    Args:
        w: The (dimensionless) frequencies at which to evaluate the power spectrum, of
            shape (n, ).
        log_rho_tilde: log(amplitude * lengthscale ** 2), can be batched with shape
            (b, ).
        log_lengthscale: The kernel log lengthscale, can be batched with shape (b, ).

    Returns:
        The RBF spectrum derivative at U, of shape (b, n).

    """
    return (
        np.exp(log_rho_tilde)
        * np.sqrt(2 * math.pi)
        * np.exp(-2 * math.pi ** 2 * np.exp(log_lengthscale) * w ** 2)
    )


def rbf_spectrum_dldl(
    w: np.ndarray, log_rho_tilde: np.ndarray, log_lengthscale: np.ndarray
) -> np.ndarray:
    """Evaluate the RBF spectrum 2nd derivative w.r.t. log_lengthscale.

    Args:
        w: The (dimensionless) frequencies at which to evaluate the power spectrum, of
            shape (n, ).
        log_rho_tilde: log(amplitude * lengthscale ** 2), can be batched with shape
            (b, ).
        log_lengthscale: The kernel log lengthscale, can be batched with shape (b, ).

    Returns:
        The RBF spectrum derivative at U, of shape (b, n).

    """
    return (
        -2
        * math.pi ** 2
        * np.sqrt(2 * math.pi)
        * w ** 2
        * np.exp(log_rho_tilde)
        * np.exp(-2 * math.pi ** 2 * np.exp(log_lengthscale) * w ** 2 + log_lengthscale)
        * (-2 * math.pi ** 2 * np.exp(log_lengthscale) * w ** 2 + 1)
    )


def rbf_spectrum_dldr(
    w: np.ndarray, log_rho_tilde: np.ndarray, log_lengthscale: np.ndarray
) -> np.ndarray:
    """Evaluate the RBF spectrum derivative w.r.t. log_rho_tilde and log_lengthscale.

    Args:
        w: The (dimensionless) frequencies at which to evaluate the power spectrum, of
            shape (n, ).
        log_rho_tilde: log(amplitude * lengthscale ** 2), can be batched with shape
            (b, ).
        log_lengthscale: The kernel log lengthscale, can be batched with shape (b, ).

    Returns:
        The RBF spectrum derivative at U, of shape (b, n).

    """
    return (
        np.exp(log_rho_tilde)
        * np.sqrt(2 * math.pi)
        * np.exp(-2 * math.pi ** 2 * np.exp(log_lengthscale) * w ** 2)
        * -2
        * math.pi ** 2
        * np.exp(log_lengthscale)
        * w ** 2
    )
