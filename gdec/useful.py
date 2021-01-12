"""A file filled with Useful Functions.

“It's a Useful Pot,” said Pooh. “Here it is. And it's got 'A Very Happy
Birthday with love from Pooh' written on it. That's what all that writing is.
And it's for putting things in. There!”

"""
from typing import Optional

import numpy as np
from scipy import special


def add_intercept_feature_col(X: np.ndarray) -> np.ndarray:
    """Add an intercept column to a (n_examples, n_features) data matrix.

    Args:
        X: The input array, of shape (n_examples, n_features)

    Returns:
        X, but with a column of ones prepended, so that it is of shape
        (n_examples, n_features + 1)

    """
    return np.insert(X, 0, 1.0, axis=1)


def log_softmax(a: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    """Compute the log softmax of an array.

    Args:
        a: The input array.
        axis: Axis or axes over which the softmax is calculated.

    Returns:
        An array of the same shape as `a`, containing the log softmax.

    """
    return a - special.logsumexp(a, axis, keepdims=True, return_sign=False)


def encode_one_hot(labels: np.ndarray, vector_size: int) -> np.ndarray:
    """Encode integer class labels as one-hot vectors.

    Args:
        labels: A 1D vector of the integer class labels.
        vector_size: The desired size of each one-hot vector.

    Returns:
        A 2D array containing the one-hot encoded vectors in the rows.

    """
    assert labels.ndim == 1
    assert vector_size > np.max(labels)
    encoded = np.zeros((labels.size, vector_size))
    encoded[np.arange(labels.size), labels] = 1
    return encoded


def categorical_sample(probs: np.ndarray) -> np.ndarray:
    """Sample from a categorical distribution.

    This is vectorized, and so faster than scipy.stats.multinomial.

    Args:
        probs: A numpy array of shape `(b0, ..., bm, m), containing m class
            probabilities along the last axis. Can also have a shape of just `(m, )`.

    Returns:
        A numpy array of shape `(b0, ..., bm)` containing the class labels for each
        sample, in `{0, ..., m - 1}`.

    """
    assert np.allclose(probs.sum(axis=-1), 1)
    unifs = np.expand_dims(np.random.random(probs.shape[:-1]), -1)
    return np.argmax(unifs <= np.cumsum(probs, axis=-1), axis=-1)


def circdist(x: np.ndarray, y: np.ndarray, circumference: float) -> np.ndarray:
    """Calculate the signed circular distance between two arrays.

    Returns positive numbers if y is clockwise compared to x, negative if y is counter-
    clockwise compared to x.

    Args:
        x: The first array.
        y: The second array.
        circumference: The circumference of the circle.

    Returns:
        An array of the same shape as x and y, containing the signed circular distances.

    """
    assert y.shape == x.shape
    return -np.mod(x - y - circumference / 2, circumference) + circumference / 2


def product(*args: np.ndarray) -> np.ndarray:
    """Compute the cartesian product of input arrays.

    Args:
        args: The input arrays, of shape `(n, )` (can have different lengths).

    Returns:
        An array containing the cartesian product of args, with the tuples in the rows.
    """
    for i in args:
        if i.ndim != 1:
            raise ValueError("Input arrays must be 1D.")
    return np.stack(np.meshgrid(*args, indexing="ij")).reshape(len(args), -1).T
