"""Super neuron decoder.

See Stringer, Carsen, Michalis Michaelos, and Marius Pachitariu. 2019. “High Precision
Coding in Mouse Visual Cortex.” bioRxiv. https://doi.org/10.1101/679324.

"""
import numpy as np
import sklearn
from sklearn import linear_model

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


class SuperNeuronDecoder(sklearn.base.BaseEstimator, sklearn.base.ClassifierMixin):
    """Super neuron decoder.

    Attributes:
        W_: When fitted, the weight matrix, of shape ``(n_classes, n_features)``.
        b_: When fitted, the intercept vector, of shape ``(n_classes, )``.
    """

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the SND.

        Args:
            X: An array of shape ``(n_samples, n_features)`` containing the training
                examples.
            y: An array of shape ``(n_samples, )`` containing the training labels.

        """
        linreg = linear_model.LinearRegression()
        linreg.fit(X, to_linear_targets(y))
        self.W_ = linreg.coef_
        self.b_ = linreg.intercept_

    def _predict_log_probs(self, X: np.ndarray) -> np.ndarray:
        sklearn.utils.validation.check_is_fitted(self)
        X = sklearn.utils.validation.check_array(X)
        scores = X @ self.W_.T + self.b_[None, :]
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
