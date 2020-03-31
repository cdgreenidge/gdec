"""Gaussian process utilities for JAX code."""
import logging
import math
from typing import Callable, Tuple

import jax.numpy as np

logger = logging.getLogger(__name__)


def choose_n_basis_funs(
    spectrum_fn: Callable[[np.ndarray], np.ndarray], threshold: float = 1.0e-3
) -> int:
    """Chooses the appropriate number of basis functions.

    Args:
        spectrum_fn: A function that evaluates the power spectrum.
        threshold: The minimum covariance value.

    """
    n_periodic = 0
    while (spectrum_fn(np.array([n_periodic])) > threshold).any():
        n_periodic += 1
    return 1 + 2 * n_periodic  # DC, n_periodic cosines, n_periodic sines


def fourier_basis(n_domain: int, n_funs: int) -> Tuple[np.ndarray, np.ndarray]:
    """Make a Fourier basis.

    Args:
        n_domain: The number of points in the domain of each basis function (i.e. the
            array size in the "IFFT".)
        n_funs: The number of basis functions. Must be odd.

    Returns:
        An array of shape `(n_domain, n_funs)` containing the basis functions, and an
        array containing the spectral covariances, of shape `(n_funs, )`.

    """
    assert n_funs % 2 != 0
    freqs = np.arange(1, (n_funs // 2) + 1)

    basis_x = (2 * math.pi / n_domain) * np.outer(np.arange(n_domain), freqs)
    cosines = 2 * np.cos(basis_x)
    sines = -2 * np.sin(basis_x)
    dc = np.ones((n_domain, 1))
    basis = np.concatenate((dc, cosines, sines), axis=1)

    spectrum_freqs = np.concatenate((np.array([0.0]), freqs, freqs))
    return (basis, spectrum_freqs)


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


def whitened_fourier_basis(
    spectrum_fn: Callable[[np.ndarray], np.ndarray], n_domain: int, n_funs: int
) -> np.ndarray:
    """Make a whitened Fourier basis.

    This is like the standard Fourier basis, but reparameterized so that the coefficient
    prior is standard normal instead of `N(0, diag(w))` where `w` is the prior
    spectral power density.

    Args:
        spectrum_fn: A function that evaluates the prior spectral density.
        n_domain: The number of points in the domain of each basis function (i.e. the
            array size in the "IFFT".)
        n_funs: The number of basis functions. Must be odd.

    Returns:
        An array of shape `(n_domain, n_funs)` containing the basis functions

    """
    basis, spectrum_freqs = fourier_basis(n_domain, n_funs)
    spectrum = spectrum_fn(spectrum_freqs)
    whitened_basis = np.sqrt(spectrum)[None, :] * basis
    return whitened_basis
