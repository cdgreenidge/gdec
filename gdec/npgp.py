"""Gaussian process utilities for Numpy code."""
import logging
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)


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
