"""Utilities."""
import logging
from typing import Optional

import numpy as np
import torch

from gdec import useful

logger = logging.getLogger(__name__)


def mean_abs_err(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    y_pred_c: float,
    y_true_c: Optional[float] = None,
) -> float:
    """Compute the mean absolute error, in degrees, of some model predictions.

    Args:
        y_pred: A vector of the predicted class labels.
        y_true: A vector of the true class labels.
        y_pred_c: The circumference of the circle of predictions, in units of class
            labels.
        y_true_c: The circumference of the circle of true values, in units of class
            labels. If `None`, is assumed to be the same as `y_pred_c`.

    Returns:
        The mean absolute prediction error, in degrees.

    """
    if y_true_c is None:
        y_true_c = y_pred_c

    pred_degrees = (360.0 / y_pred_c) * y_pred
    true_degrees = (360.0 / y_true_c) * y_true
    dists = useful.circdist(pred_degrees, true_degrees, 360.0)
    return np.mean(np.abs(dists))


def jitter(K: torch.Tensor, jitter: float = 1.0e-9) -> torch.Tensor:
    """Add jitter to the diagonal of a matrix.

    Args:
        K: The matrix to which to add the jitter. Should be two-dimensional.
        jitter: The amount of jitter to add.

    Returns:
        The jittered matrix.

    """
    assert K.dim() == 2
    return K + torch.diag(
        torch.full((K.size()[0],), jitter, dtype=K.dtype, device=K.device)
    )
