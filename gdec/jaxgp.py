"""Gaussian process utilities for Numpy code."""
import math
from typing import Tuple

import jax.numpy as np


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
    freqs = np.concatenate(
        [np.array([dc_freq]), np.array(cosine_freqs), np.array(sine_freqs[::-1])]
    )

    return basis / np.sqrt(n), freqs / n


def rbf_spectrum(
    w: np.ndarray, amplitude: np.ndarray, lengthscale: np.ndarray
) -> np.ndarray:
    """Evaluate the RBF spectrum element-wise at ``w``.

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
    """Evaluate the RBF spectrum derivative w.r.t. amplitude.

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
    """Evaluate the RBF spectrum derivative w.r.t. lengthscale.

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
    """Evaluate the RBF spectrum 2nd derivative w.r.t. amplitude.

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
    """Evaluate the RBF spectrum 2nd derivative w.r.t. lengthscale.

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
    """Evaluate the RBF spectrum derivative w.r.t. amplitude and lengthscale..

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
