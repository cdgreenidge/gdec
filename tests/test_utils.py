"""Test utils.py."""
import numpy as np

from gdec import utils


def test_mean_abs_error_calculates_the_mean_abs_err():
    circumference = 4
    true = np.array([0, 3, 1, 1])
    est = np.array([0, 0, 0, 2])
    assert utils.mean_abs_err(true, est, circumference) == 67.5
