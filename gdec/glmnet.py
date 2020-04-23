"""Regularized multinomial logistic regression."""
import numpy as np
from sklearn import linear_model


class LogisticRegression(linear_model.LogisticRegressionCV):
    """Logistic regression.

    A simple pass-through of the linear_model.LogisticRegressionCV class, but with
    a utility method to enable easier downstream analysis.

    """

    @property
    def coefs_(self) -> np.ndarray:
        """Return the model coefficients."""
        return self.coef_
