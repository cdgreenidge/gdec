"""Test snd.py."""
from sklearn import model_selection

import gdec
from gdec import synthetic


def test_you_can_train_the_snd_on_the_synthetic_dataset():
    X, y = synthetic.generate_dataset(seed=1686, n_classes=32)
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, train_size=0.8
    )
    model = gdec.SuperNeuronDecoder()
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    assert score > 1 / 32  # Better than random guessing?


def test_you_can_train_the_snd_with_intercept():
    X, y = synthetic.generate_dataset(seed=1686, n_classes=32)
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, train_size=0.8
    )
    model = gdec.SuperNeuronDecoder(affine=True)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    assert score > 1 / 32  # Better than random guessing?
