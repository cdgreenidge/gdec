"""Test synthetic.py."""
from sklearn import linear_model, model_selection

from gdec import synthetic


def test_you_can_train_logistic_regression_on_the_synthetic_dataset():
    X, y = synthetic.generate_dataset(seed=1634, n_classes=32)
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, train_size=0.8
    )
    model = linear_model.LogisticRegression(solver="newton-cg")
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    assert score > 1 / 32  # Better than random guessing?
