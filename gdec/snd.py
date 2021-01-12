"""Super neuron decoder.

See Stringer, Carsen, Michalis Michaelos, and Marius Pachitariu. 2019. “High Precision
Coding in Mouse Visual Cortex.” bioRxiv. https://doi.org/10.1101/679324.

"""
from typing import Any

import numpy as np
import sklearn
from scipy import stats
from sklearn import preprocessing

from gdec import useful


def to_linear_targets(y: np.ndarray) -> np.ndarray:
    """Transform a vector of classes into a linear regression target matrix.

    Args:
        y: The vector of classes. The classes should be in (0, ..., k - 1)
            where there are k classes. Of shape (n, ).

    Returns:
        A matrix of shape (n, k) where each column contains the von-Mises tuning
        function evaluated at the degree value represented by the class labels.
        Each column has a different preferred angle, evenly tiling the unit circle from
        0 to 360 degrees.

    """
    classes, inverse = np.unique(y, return_inverse=True)
    assert (classes == range(len(classes))).all()
    radians = classes * (2 * np.pi / len(classes))
    y_rad = radians[inverse]
    theta_pref = np.linspace(0, 2 * np.pi, num=len(classes), endpoint=False)
    sigma = 0.1
    targets = np.exp(np.cos(y_rad[:, None] - theta_pref[None, :]) / sigma)
    return targets / np.max(targets, axis=0)  # Normalize


def fast_ridge(X: np.ndarray, y: np.ndarray, lam: float = 1.0) -> np.ndarray:
    """Fast ridge regression, the original fast_ridge code from Stringer et. al.

    Args:
        X: A matrix of shape (n_examples, n_features).
        y: a vector of shape (n_examples, n_targets).

    Returns:
        A coefficient matrix of shape (n_features, n_targets).

    """
    X = X.T
    # We add the above line because Stringer et. al's code uses a matrix with
    # (n_features, n_examples) layout

    N, M = X.shape
    lam = lam * M
    if N < M:
        XXt = X @ X.T
        w = np.linalg.solve(XXt + lam * np.eye(N), X @ y)
    else:
        XtX = X.T @ X
        w = 1 / lam * X @ (y - np.linalg.solve(lam * np.eye(M) + XtX, XtX @ y))
    return w


class SuperNeuronDecoder(sklearn.base.BaseEstimator, sklearn.base.ClassifierMixin):
    """Super neuron decoder.

    Args:
        affine: whether or not to include a linear intercept term.

    Attributes:
        coefs_: When fitted, the weight matrix, of shape ``(n_classes, n_features)``.
        intercept_: When fitted, the intercept vector, of shape ``(n_classes, )``.

    """

    def __init__(self, *args: Any, affine: bool = False, **kwargs: Any) -> None:
        self.affine = affine
        super().__init__(*args, **kwargs)

    def fit(self, X: np.ndarray, y: np.ndarray, intercept: bool = False) -> None:
        """Fit the SND.

        Args:
            X: An array of shape ``(n_samples, n_features)`` containing the training
                examples.
            y: An array of shape ``(n_samples, )`` containing the training labels.

        """
        X = preprocessing.scale(X.astype(np.float64), axis=0)
        if self.affine:
            X = useful.add_intercept_feature_col(X)
        y = to_linear_targets(y)
        # No idea why zscore is necessary, but it's in Stringer's code
        y = stats.zscore(y, axis=1)
        self.coefs_ = fast_ridge(X, y).T

    def _predict_log_probs(self, X: np.ndarray) -> np.ndarray:
        sklearn.utils.validation.check_is_fitted(self)
        X = sklearn.utils.validation.check_array(X)
        if self.affine:
            X = useful.add_intercept_feature_col(X)
        scores = X @ self.coefs_.T
        return useful.log_softmax(scores, axis=-1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.

        Args:
            X: An array of shape ``(n_samples, n_features)`` containing the examples.

        Returns:
            An array of shape ``(n_samples, n_classes)`` containing the predicted
            probabilities for each class.

        """
        return np.exp(self._predict_log_probs(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts class labels for X.

        Args:
            X: An array of shape ``(n_samples, n_features)`` containing the training
                examples.

        Returns:
            An array of shape ``(n_samples, )`` containing the predicted labels.

        """
        log_probs = self._predict_log_probs(X)
        return np.argmax(log_probs, axis=1)
