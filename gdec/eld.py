"""Empirical Linear Decoder.

See Graf, Arnulf B. A., Adam Kohn, Mehrdad Jazayeri, and J. Anthony Movshon. 2011.
“Decoding the Activity of Neuronal Populations in Macaque Primary Visual Cortex.” Nature
Neuroscience 14 (2): 239–45.

"""
import warnings
from typing import List, Tuple

import jax.numpy as jnp
import jax.scipy.special
import numpy as np
import scipy.optimize
import sklearn.base
import sklearn.model_selection
import sklearn.preprocessing
import sklearn.svm
import sklearn.utils.validation


def circdist(x: jnp.ndarray, y: jnp.ndarray, circumference: float) -> jnp.ndarray:
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
    return -jnp.mod(x - y - circumference / 2, circumference) + circumference / 2


def logsoftmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Apply log softmax to an array of logits, log-normalizing along an axis.

    Args:
        x: The array of logits.
        axis: The axis along which to normalize.

    Returns:
        The log-softmax of x.

    """
    return x - jax.scipy.special.logsumexp(x, axis, keepdims=True)


def split_dataset(X: np.ndarray, y: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Split a dataset with ``k`` orientations into ``k - 1`` 1v1 datasets.

    Each sub-dataset has the ``ith`` and the ``i-1th`` orientation.

    Args:
        X: An array of shape ``(n_samples, n_features)`` containing the training
            examples.
        y: An array of shape ``(n_samples, )`` containing the training labels.

    Returns:
        A list of tuples, each containing the (X, y) pair for the ith subset.

    """
    datasets: List[Tuple[np.ndarray, np.ndarray]] = []
    classes = np.unique(y)
    np.sort(classes)
    for i in range(1, len(classes)):
        y_0 = classes[i - 1]
        y_1 = classes[i]
        mask = np.logical_or(y == y_0, y == y_1)
        X_sub = X[mask, :]
        y_sub = (y[mask] == y_1).astype(np.int)
        datasets.append((X_sub, y_sub))
    return datasets


def svm_hyperplane(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Train an SVM on a dataset and return the hyperplane parameters.

    The vector and offset are scaled so that the vector norm is 1. The slack penalty
    is set by cross-validation.

    Args:
        X: An array of shape ``(n_samples, n_features)`` containing the training
            examples.
        y: An array of shape ``(n_samples, )`` containing the training labels.

    Returns:
        A tuple containing ``w``, the normal vector to the decision plane, and ``b``,
        the offset.

    """
    params = {"C": np.logspace(-2, 1, num=10)}
    cv = sklearn.model_selection.StratifiedShuffleSplit(n_splits=3, train_size=0.8)
    model = sklearn.model_selection.GridSearchCV(
        sklearn.svm.LinearSVC(max_iter=1e6), params, cv=cv
    )
    model.fit(X, y)
    w_norm = np.linalg.norm(model.best_estimator_.coef_)
    w = (model.best_estimator_.coef_ / w_norm).squeeze()
    b = model.best_estimator_.intercept_ / w_norm
    return w, b


def eld_transform(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the ELD data transform.

    Args:
        X: An array of shape ``(n_samples, n_features)`` containing the training
            examples.
        y: An array of shape ``(n_samples, )`` containing the training labels.

    Returns:
        A matrix ``W`` containing the SVM hyperplane normal vectors in each row, and a
        vector ``b`` containing the offsets.

    """
    datasets = split_dataset(X, y)
    n_orientations = len(datasets) + 1
    n_features = datasets[0][0].shape[1]
    W = np.zeros((n_orientations, n_features))
    b = np.zeros(n_orientations)
    for i, (X, y) in enumerate(datasets):
        W[i + 1, :], b[i + 1] = svm_hyperplane(X, y)
    return W, b


class EmpiricalLinearDecoder(sklearn.base.BaseEstimator, sklearn.base.ClassifierMixin):
    """Empirical linear decoder."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the empirical linear decoder.

        Args:
            X: An array of shape ``(n_samples, n_features)`` containing the training
                examples.
            y: An array of shape ``(n_samples, )`` containing the training labels.
            criterion: A string, one of "squared_circdist" or "cross_entropy", denoting
                the criterion minimized to choose the hyperparameters.

        """
        sklearn.utils.validation.check_X_y(X, y)
        self.classes_ = np.unique(y)
        np.sort(self.classes_)
        self.scaler_ = sklearn.preprocessing.MaxAbsScaler()
        self.scaler_.fit(X)
        X_scaled = self.scaler_.transform(X)

        self.coefs_, self.b_ = eld_transform(X_scaled, y)

        def cross_entropy(alpha: np.ndarray) -> np.ndarray:
            """Cross-entropy loss function for fitting alpha hyperparams."""
            log_probs = self._log_probs(X_scaled, alpha)
            return -jnp.take_along_axis(log_probs, y[:, None], axis=1).sum()

        def squared_circdist(alpha: np.ndarray) -> np.ndarray:
            """Squared circular distance loss function for fitting alpha hyperparams."""
            predictions = jnp.exp(self._log_probs(X_scaled, alpha)) @ self.classes_
            return (circdist(predictions, y, self.classes_.size) ** 2).sum()

        grad_loss = jax.jit(jax.grad(cross_entropy))
        hess_loss = jax.jit(jax.hessian(cross_entropy))

        alpha_0 = jnp.ones(self.coefs_.shape[0])
        opt_results = scipy.optimize.minimize(
            cross_entropy, alpha_0, method="trust-ncg", jac=grad_loss, hess=hess_loss
        )
        if not opt_results.success:
            warnings.warn(opt_results.message, UserWarning)
        self.alpha_ = opt_results.x

    def _log_probs(self, X: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        """Compute class log probabilities for X.

        Args:
            X: An array of shape ``(n_samples, n_features)`` containing the training
                examples.
            alpha: The SVM normal vector scales. Normally this should be
                ``self.alpha_``, but we leave it as an argument so we can differentiate
                through this function when fitting.

        Returns:
            An array of shape ``(n_samples, n_classes)`` containing the predicted log
            probabilities for each class.

        """
        n = alpha.shape[0]
        L = jnp.tril(np.ones((n, n)))
        A = jnp.diag(alpha)
        likelihoods = (L @ A @ (self.coefs_ @ X.T + self.b_[:, None])).T
        return logsoftmax(likelihoods)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predicts class probabilities for X.

        Args:
            X: An array of shape ``(n_samples, n_features)`` containing the training
                examples.

        Returns:
            An array of shape ``(n_samples, n_class)`` containing the predicted
            probabilities.

        """
        X_scaled = self.scaler_.transform(X)
        return np.exp(self._log_probs(X_scaled, self.alpha_))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts class labels for X.

        Args:
            X: An array of shape ``(n_samples, n_features)`` containing the training
                examples.

        Returns:
            An array of shape ``(n_samples, )`` containing the predicted labels.

        """
        sklearn.utils.validation.check_is_fitted(self)
        X = sklearn.utils.validation.check_array(X)
        X_scaled = self.scaler_.transform(X)
        indices = np.asarray(np.argmax(self._log_probs(X_scaled, self.alpha_), axis=1))
        return self.classes_[indices]
