"""Test eld.py."""
import numpy as np
from sklearn import model_selection

import gdec
from gdec import eld, synthetic


def test_circdist_gives_correct_distances():
    c = 12
    x = 8 * np.ones((12,))
    y = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    actual = eld.circdist(x, y, c)
    expected = [4, 5, 6, -5, -4, -3, -2, -1, 0, 1, 2, 3]
    assert np.array_equal(actual, expected)


def test_you_can_train_the_eld_on_the_synthetic_dataset():
    X, y = synthetic.generate_dataset(seed=1686, n_classes=32)
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, train_size=0.8
    )
    model = gdec.EmpiricalLinearDecoder()
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    assert score > 1 / 32  # Better than random guessing?
