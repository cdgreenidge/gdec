"""Gaussian process utilities for Torch code."""
import math
from typing import Callable, Tuple

import torch


def choose_n_basis_funs(
    spectrum_fn: Callable[[torch.Tensor], torch.Tensor], threshold: float = 1.0e-3
) -> int:
    """Chooses the appropriate number of basis functions.

    Args:
        spectrum_fn: A function that evaluates the power spectrum.
        threshold: The minimum covariance value.

    """
    n_periodic = 0
    while (spectrum_fn(torch.tensor(n_periodic)) > threshold).any():
        n_periodic += 1
    return 1 + 2 * n_periodic  # DC, n_periodic cosines, n_periodic sines


def make_basis(n_domain: int, n_funs: int) -> Tuple[torch.Tensor, torch.Tensor]:
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
    freqs = torch.arange(1, (n_funs // 2) + 1, dtype=torch.float)

    basis_x = (2 * math.pi / n_domain) * torch.ger(
        torch.arange(n_domain, dtype=torch.float), freqs
    )
    cosines = 2 * torch.cos(basis_x)
    sines = -2 * torch.sin(basis_x)
    dc = torch.ones((n_domain, 1))
    basis = torch.cat((dc, cosines, sines), dim=1)

    spectrum_freqs = torch.cat((torch.tensor([0.0]), freqs, freqs))
    return (basis, spectrum_freqs)


def matern_5_2_spectrum(
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
        * (torch.tensor(400 * math.sqrt(5)) / (3 * lengthscales ** 2))
        * ((torch.tensor(5.0) / lengthscales ** 2) + 4 * math.pi ** 2 * w ** 2) ** (-3)
    )
