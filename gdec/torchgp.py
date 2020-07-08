"""Gaussian process utilities for Torch code."""
import math
from typing import Tuple

import torch


def real_fourier_basis(n: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Make a Fourier basis.

    Args:
        n: The basis size

    Returns:
        An array of shape `(n_domain, n_funs)` containing the basis functions, and an
        array containing the spectral covariances, of shape `(n_funs, )`.

    """
    assert n > 1
    dc = torch.ones((n,))
    dc_freq = 0

    cosine_basis_vectors = []
    cosine_freqs = []
    sine_basis_vectors = []
    sine_freqs = []

    ts = torch.arange(n)
    for w in range(1, 1 + (n - 1) // 2):
        x = w * (2 * math.pi / n) * ts
        cosine_basis_vectors.append(math.sqrt(2) * torch.cos(x))
        cosine_freqs.append(w)
        sine_basis_vectors.append(-math.sqrt(2) * torch.sin(x))
        sine_freqs.append(w)

    if n % 2 == 0:
        w = n // 2
        x = w * 2 * math.pi * ts / n
        cosine_basis_vectors.append(torch.cos(x))
        cosine_freqs.append(w)

    basis = torch.stack((dc, *cosine_basis_vectors, *sine_basis_vectors[::-1]), -1)
    freqs = torch.cat(
        (
            torch.tensor([dc_freq], dtype=torch.float),
            torch.tensor(cosine_freqs, dtype=torch.float),
            torch.tensor(sine_freqs[::-1], dtype=torch.float),
        )
    )

    return basis / math.sqrt(n), freqs / n


def rbf_spectrum(
    w: torch.Tensor, amplitudes: torch.Tensor, lengthscales: torch.Tensor
) -> torch.Tensor:
    """Evaluate the Matern 5/2 power spectrum element-wise at ``w``.

    Args:
        w: The (dimensionless) frequencies at which to evaluate the power spectrum, of
            shape (n, ).
        amplitude: The kernel amplitude, can be batched with shape (b, ).
        lengthscale: The kernel lengthscale, can be batched with shape (b, ).

    Returns:
        The Matern 5/2 spectrum evaluated at U, of shape (b, n) (if the input amplitudes
        and lengthscales have a batch dimension) and shape (n, ) otherwise.

    """
    amplitudes = torch.unsqueeze(amplitudes, -1)
    lengthscales = torch.unsqueeze(lengthscales, -1)
    return (
        amplitudes ** 2
        * torch.sqrt(2 * math.pi * lengthscales ** 2)
        * torch.exp(-2 * math.pi ** 2 * lengthscales ** 2 * w ** 2)
    )


def matern_spectrum(
    w: torch.Tensor,
    amplitudes: torch.Tensor,
    lengthscales: torch.Tensor,
    nu: float = 1.5,
) -> torch.Tensor:
    """Evaluate the Matern 5/2 power spectrum element-wise at ``w``.

    Args:
        w: The (dimensionless) frequencies at which to evaluate the power spectrum, of
            shape (n, ).
        amplitude: The kernel amplitude, can be batched with shape (b, ).
        lengthscale: The kernel lengthscale, can be batched with shape (b, ).
        nu: The smoothness parameter.

    Returns:
        The Matern 5/2 spectrum evaluated at U, of shape (b, n) (if the input amplitudes
        and lengthscales have a batch dimension) and shape (n, ) otherwise.

    """
    amplitudes = torch.unsqueeze(amplitudes, -1)
    lengthscales = torch.unsqueeze(lengthscales, -1)
    return (
        amplitudes ** 2
        * (2 * math.sqrt(math.pi) * math.gamma(nu + 0.5) * (2 * nu) ** nu)
        / (math.gamma(nu) * lengthscales ** (2 * nu))
        * (2 * nu / lengthscales ** 2 + 4 * math.pi ** 2 * w ** 2) ** -(nu + 0.5)
    )
