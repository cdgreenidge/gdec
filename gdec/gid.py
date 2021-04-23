"""Gaussian independent decoder (i.e., Naive Bayes)."""
import numpy as np
import sklearn.utils
from sklearn import naive_bayes


class GaussianIndependentDecoder(naive_bayes.GaussianNB):
    """Gaussian independent decoder.

    A simple pass-through of the naive_bayes.GaussianNB class, but with
    a utility method to enable easier downstream analysis.

    """

    @property
    def coefs_(self) -> np.ndarray:
        """Return the model coefficients."""
        return self.theta_


class LinearGaussianIndependentDecoder(naive_bayes.GaussianNB):
    """Gaussian independent decoder, but with a linear decision boundary.

    This happens when the variance is fixed over classes. Note: partial fitting is
    not implemented.

    """

    @property
    def coefs_(self) -> np.ndarray:
        """Return the model coefficients."""
        return self.theta_

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearGaussianIndependentDecoder":
        """Fit Poisson Naive Bayes according to X, y.

        Args:
            X: An array of shape ``(n_samples, n_features)`` containing the training
                examples.
            y: An array of shape ``(n_samples, )`` containing the training labels.

        Returns:
            self, an object.

        Raises:
            ValueError: if X is not an array with int/uint dtype, since the Poisson PDF
                is only defined on the integers.

        """
        X, y = sklearn.utils.check_X_y(X, y)

        self.classes_ = np.sort(np.unique(y))
        n_features = X.shape[1]
        n_classes = len(self.classes_)
        self.theta_ = np.zeros((n_classes, n_features))  # Firing means
        self.sigma_ = np.zeros((n_classes, n_features))  # Firing means
        self.class_count_ = np.zeros(n_classes)

        for i, y_i in enumerate(self.classes_):
            i = self.classes_.searchsorted(y_i)
            X_i = X[y == y_i, :]
            self.coefs_[i, :] = np.mean(X_i, axis=0)
            self.class_count_[i] = X_i.shape[0]

        self.sigma_ = np.tile(np.var(X, axis=0), (n_classes, 1))
        self.class_prior_ = self.class_count_ / self.class_count_.sum()

        return self

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        """Partial fit--not implemented."""
        raise NotImplementedError  # Make sure no-one calls partial fit on accident
