"""A file filled with Useful Functions.

“It's a Useful Pot,” said Pooh. “Here it is. And it's got 'A Very Happy
Birthday with love from Pooh' written on it. That's what all that writing is.
And it's for putting things in. There!”

"""
from typing import Optional

import numpy as np
from scipy import special


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
        probs: A numpy array of shape `(b0, ..., bm, m), containing m class probabilities
            along the last axis. Can also have a shape of just `(m, )`.

    Returns:
        A numpy array of shape `(b0, ..., bm)` containing the class labels for each
        sample, in `{0, ..., m - 1}`.

    """
    assert np.allclose(probs.sum(axis=-1), 1)
    unifs = np.expand_dims(np.random.random(probs.shape[:-1]), -1)
    return np.argmax(unifs <= np.cumsum(probs, axis=-1), axis=-1)
