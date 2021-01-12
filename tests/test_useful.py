"""Test useful.py."""
import itertools

import hypothesis
import numpy as np
from hypothesis import strategies
from hypothesis.extra import numpy
from scipy import special

from gdec import useful

array_strat = numpy.arrays(
    float,
    numpy.array_shapes(),
    elements=strategies.floats(
        min_value=1e9, max_value=1e9, allow_nan=False, allow_infinity=False
    ),
)


@strategies.composite
def array_and_axis(draw):
    array = draw(array_strat)
    axis = draw(strategies.integers(min_value=0, max_value=array.ndim - 1))
    return array, axis


@hypothesis.given(
    numpy.arrays(
        float,
        numpy.array_shapes(min_dims=2, max_dims=2),
        elements=strategies.floats(
            min_value=1e9, max_value=1e9, allow_nan=False, allow_infinity=False
        ),
    )
)
def test_add_intercept_feature_col_prepends_a_col_of_ones(X):
    X1 = useful.add_intercept_feature_col(X)
    np.testing.assert_array_almost_equal(X1[:, 0], 1)
    np.testing.assert_array_almost_equal(X1[:, 1:], X)


@hypothesis.given(array_and_axis())
def test_log_softmax_normalizes_to_1_along_selected_axis(x):
    array, axis = x
    softmax = np.exp(useful.log_softmax(array, axis=axis))
    np.testing.assert_array_almost_equal(softmax.sum(axis), 1)


@hypothesis.given(array_and_axis())
def test_log_softmax_is_equal_to_naive_implementation(x):
    array, axis = x
    log_softmax = useful.log_softmax(array, axis=axis)
    naive = np.log(special.softmax(array, axis))
    np.testing.assert_array_almost_equal(log_softmax, naive)


@strategies.composite
def labels_and_vector_size(draw):
    vector_size = draw(strategies.integers(min_value=2, max_value=32))
    labels = draw(
        numpy.arrays(
            int,
            numpy.array_shapes(max_dims=1),
            elements=strategies.integers(0, vector_size - 1),
        )
    )
    return labels, vector_size


@hypothesis.given(labels_and_vector_size())
def test_one_hot_returns_a_one_hot_encoding(x):
    labels, vector_size = x
    one_hot = useful.encode_one_hot(labels, vector_size)
    for i, label in enumerate(labels):
        assert one_hot[i][label] == 1
        assert (np.delete(one_hot[i], label) == 0).all()


def test_categorical_sample_samples_with_the_correct_probabilities():
    probs = np.array([0.0, 1.0, 0.0])
    for _ in range(3):
        sample = useful.categorical_sample(probs)
        assert sample.item() == 1


@hypothesis.given(labels_and_vector_size())
def test_categorical_samples_vectorized_probabilities(x):
    labels, vector_size = x
    probs = useful.encode_one_hot(labels, vector_size)
    for _ in range(3):
        sample = useful.categorical_sample(probs)
        assert (sample == labels).all()


def test_circdist_gives_correct_distances():
    c = 12
    x = 8 * np.ones((12,))
    y = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    actual = useful.circdist(x, y, c)
    expected = [4, 5, 6, -5, -4, -3, -2, -1, 0, 1, 2, 3]
    assert np.array_equal(actual, expected)


array_strat_1d = numpy.arrays(
    float,
    numpy.array_shapes(min_dims=1, max_dims=1, min_side=1, max_side=8),
    elements=strategies.floats(
        min_value=-100, max_value=100, allow_nan=False, allow_infinity=False
    ),
)


@strategies.composite
def random_number_of_1d_arrays(draw):
    n_arrays = draw(strategies.integers(min_value=1, max_value=3))
    arrays = [draw(array_strat_1d) for _ in range(n_arrays)]
    return arrays


@hypothesis.given(random_number_of_1d_arrays())
def test_cartesian_product_returns_cartesian_product_of_arrays(x):
    product = useful.product(*x)
    product_tuples = list(itertools.product(*x))
    assert len(product_tuples) == product.shape[0]
    for row in product:
        assert tuple(row) in product_tuples
