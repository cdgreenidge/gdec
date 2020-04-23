"""Poisson independent decoder (i.e. Naive Bayes with Poisson likelihood)."""
import numpy as np
import scipy.stats
import sklearn.naive_bayes
import sklearn.utils
import sklearn.utils.validation


class PoissonIndependentDecoder(sklearn.naive_bayes._BaseDiscreteNB):
    """Poisson independent decoder.

    This is just a Naive Bayes classifier with a Poisson likelihood for the features.

    Attributes:
        coefs_: When fitted, the weight matrix, of shape ``(n_classes, n_features)``.

    """

    def _joint_log_likelihood(self, X: np.ndarray) -> np.ndarray:
        """Compute the unnormalized posterior log probability of X.

        Args:
            X: An array of shape ``(n_samples, n_features)`` containing the examples.

        Returns
            An array of shape ``(n_samples, n_classes)``, containing ``log P(c) +
            log P(x|c)`` for all rows of X.

        """
        sklearn.utils.validation.check_is_fitted(self)
        X = sklearn.utils.check_array(X)

        joint_log_likelihood = np.zeros((len(self.classes_), X.shape[0]))
        for i in range(len(self.classes_)):
            mean = self.coefs_[i]
            log_prior = np.log(self.class_prior_[i])
            joint_log_likelihood[i, :] = (
                scipy.stats.poisson.logpmf(X, mean).sum(axis=1) + log_prior
            )

        return joint_log_likelihood.T

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PoissonIndependentDecoder":
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
        self.coefs_ = np.zeros((n_classes, n_features))  # Firing means
        self.class_count_ = np.zeros(n_classes)

        for i, y_i in enumerate(self.classes_):
            i = self.classes_.searchsorted(y_i)
            X_i = X[y == y_i, :]
            self.coefs_[i, :] = np.mean(X_i, axis=0)
            self.class_count_[i] = X_i.shape[0]

        self.class_prior_ = self.class_count_ / self.class_count_.sum()

        return self
