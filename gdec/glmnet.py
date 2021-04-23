"""Regularized multinomial logistic regression."""
import numpy as np
from sklearn import linear_model, pipeline, preprocessing


class LogisticRegression(pipeline.Pipeline):
    """Logistic regression.

    A simple pass-through of the linear_model.LogisticRegressionCV class, but with
    a utility method to enable easier downstream analysis.

    """

    def __init__(self, affine: bool = False) -> None:
        self.affine = affine
        super().__init__(
            [
                ("scaler", preprocessing.MaxAbsScaler(),),
                (
                    "glmnet",
                    linear_model.LogisticRegressionCV(
                        Cs=np.logspace(-4, 1, num=5),
                        solver="lbfgs",
                        cv=3,
                        n_jobs=4,
                        fit_intercept=affine,
                        multi_class="multinomial",
                    ),
                ),
            ],
        )

    @property
    def coefs_(self) -> np.ndarray:
        """Return the model coefficients."""
        return self.named_steps["glmnet"].coef_
