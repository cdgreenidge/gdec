"""Gaussian process utilities for Numpy code."""
from typing import Tuple

import numpy as np
from scipy import special


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


def expit_d(x: np.ndarray) -> np.ndarray:
    y = special.expit(x)
    return y * (1 - y)


def expit_dd(c: np.ndarray) -> np.ndarray:
    y = special.expit(c)
    return y * (1 - y) * (1 - 2 * y)


A_MIN = 0.0
A_MAX = 1e8


def g_a(c: np.ndarray) -> np.ndarray:
    return (A_MAX - A_MIN) * special.expit(c) + A_MIN


def g_a_d(c: np.ndarray) -> np.ndarray:
    return (A_MAX - A_MIN) * expit_d(c)


def g_a_dd(c: np.ndarray) -> np.ndarray:
    return (A_MAX - A_MIN) * expit_dd(c)


def g_a_inv(a: np.ndarray) -> np.ndarray:
    return special.logit((a - A_MIN) / (A_MAX - A_MIN))


B_MIN = 0.0
B_MAX = 1e8


def g_b(d: np.ndarray) -> np.ndarray:
    return (B_MAX - B_MIN) * special.expit(d) + B_MIN


def g_b_d(d: np.ndarray) -> np.ndarray:
    return (B_MAX - B_MIN) * expit_d(d)


def g_b_dd(d: np.ndarray) -> np.ndarray:
    return (B_MAX - B_MIN) * expit_dd(d)


def g_b_inv(b: np.ndarray) -> np.ndarray:
    return special.logit((b - B_MIN) / (B_MAX - B_MIN))


def rbf_natural_to_unconstrained(
    amp: np.ndarray, len_: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    return g_a_inv(amp ** 2 * len_), g_b_inv(len_ ** 2)


def rbf_unconstrained_to_natural(
    c: np.ndarray, d: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    a = g_a(c)
    b = g_b(d)
    len_ = np.sqrt(b)
    amp = np.sqrt(a / len_)
    return amp, len_


def rbf_spectrum(w: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Evaluate the RBF spectrum.

    Args:
        w: The (dimensionless) frequencies at which to evaluate the power spectrum, of
            shape (n, ).
        c: An unconstrained parameter representing
            g_a(c) = a = amplitude ** 2 * lengthscale, can be batched with shape (r, ).
        d: An unconstrained parameter representing g_b(d) = b = lengthscale ** 2, can be
            batched with shape (r, )

    Returns:
        The RBF spectrum derivative at U, of shape (r, n).

    """
    return g_a(c) * np.sqrt(2 * np.pi) * np.exp(-2 * np.pi ** 2 * g_b(d) * w ** 2)


def rbf_spectrum_dc(w: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Evaluate the RBF spectrum derivative w.r.t. c.

    Args:
        w: The (dimensionless) frequencies at which to evaluate the power spectrum, of
            shape (n, ).
        c: An unconstrained parameter representing
            g_a(c) = a = amplitude ** 2 * lengthscale, can be batched with shape (r, ).
        d: An unconstrained parameter representing g_b(d) = b = lengthscale ** 2, can be
            batched with shape (r, )

    Returns:
        The RBF spectrum derivative at U, of shape (r, n).

    """
    return g_a_d(c) * np.sqrt(2 * np.pi) * np.exp(-2 * np.pi ** 2 * g_b(d) * w ** 2)


def rbf_spectrum_dd(w: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Evaluate the RBF spectrum derivative w.r.t. d.

    Args:
        w: The (dimensionless) frequencies at which to evaluate the power spectrum, of
            shape (n, ).
        c: An unconstrained parameter representing
            g_a(c) = a = amplitude ** 2 * lengthscale, can be batched with shape (r, ).
        d: An unconstrained parameter representing g_b(d) = b = lengthscale ** 2, can be
            batched with shape (r, ).

    Returns:
        The RBF spectrum derivative at U, of shape (r, n).

    """
    return (
        g_a(c)
        * np.sqrt(2 * np.pi)
        * np.exp(-2 * np.pi ** 2 * g_b(d) * w ** 2)
        * (-2 * np.pi ** 2 * w ** 2)
        * g_b_d(d)
    )


def rbf_spectrum_dcdc(w: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Evaluate the 2nd RBF spectrum derivative w.r.t. c.

    Args:
        w: The (dimensionless) frequencies at which to evaluate the power spectrum, of
            shape (n, ).
        c: An unconstrained parameter representing
            g_a(c) = a = amplitude ** 2 * lengthscale, can be batched with shape (r, ).
        d: An unconstrained parameter representing g_b(d) = b = lengthscale ** 2, can be
            batched with shape (r, )

    Returns:
        The RBF spectrum derivative at U, of shape (r, n).

    """
    return g_a_dd(c) * np.sqrt(2 * np.pi) * np.exp(-2 * np.pi ** 2 * g_b(d) * w ** 2)


def rbf_spectrum_dcdd(w: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Evaluate the RBF spectrum cross derivative w.r.t. c and d.

    Args:
        w: The (dimensionless) frequencies at which to evaluate the power spectrum, of
            shape (n, ).
        c: An unconstrained parameter representing
            g_a(c) = a = amplitude ** 2 * lengthscale, can be batched with shape (r, ).
        d: An unconstrained parameter representing g_b(d) = b = lengthscale ** 2, can be
            batched with shape (r, )

    Returns:
        The RBF spectrum derivative at U, of shape (r, n).

    """
    return (
        g_a_d(c)
        * np.sqrt(2 * np.pi)
        * np.exp(-2 * np.pi ** 2 * g_b(d) * w ** 2)
        * (-2 * np.pi ** 2 * g_b_d(d) * w ** 2)
    )


def rbf_spectrum_dddd(w: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Evaluate the RBF spectrum 2nd derivative w.r.t. d.

    Args:
        w: The (dimensionless) frequencies at which to evaluate the power spectrum, of
            shape (n, ).
        c: An unconstrained parameter representing
            g_a(c) = a = amplitude ** 2 * lengthscale, can be batched with shape (r, ).
        d: An unconstrained parameter representing g_b(d) = b = lengthscale ** 2, can be
            batched with shape (r, )

    Returns:
        The RBF spectrum derivative at U, of shape (r, n).

    """
    return (
        g_a(c)
        * np.sqrt(2 * np.pi)
        * np.exp(-2 * np.pi ** 2 * g_b(d) * w ** 2)
        * (-2 * np.pi ** 2 * w ** 2)
        * (g_b_dd(d) - 2 * np.pi ** 2 * w ** 2 * g_b_d(d) ** 2)
    )
