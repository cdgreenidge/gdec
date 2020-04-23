"""Gaussian independent decoder (i.e., Naive Bayes)."""
import numpy as np
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
