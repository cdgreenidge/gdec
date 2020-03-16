"""Synthetic dataset generation."""
import math
from typing import Tuple

import numpy as np
from scipy import stats


def generate_dataset(
    curve_max: float = 2,
    examples_per_class: int = 32,
    kappa: float = 5,
    n_classes: int = 32,
    n_features: int = 16,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic dataset for testing decoders.

    Each neuron has a Von Mises bump tuning curve, and they are randomly correlated
    together.

    Args:
        curve_max: The maximum value of the tuning curve (firing rate) for each neuron.
        examples_per_class: The number of examples per class.
        kappa: The shape parameter of the Von Mises bump tuning curves. Higher is
            sharper.
        n_classes: The number of classes in the dataset.
        n_features: The number of features (neurons) in the dataset.
        seed: The random seed.

    Returns:
        A tuple (X, y) of examples, of shape (n_data, n_features) and labels, of shape
        (n_data, ). The features are integers representing synthesized spike counts.
        The labels are integers in {0, ..., n_classes - 1}.

    """
    np.random.seed(seed)
    classes = np.arange(n_classes)

    classes_rad = (classes / n_classes) * 2 * math.pi
    feature_tuning_locs = np.linspace(classes_rad[0], classes_rad[-1], num=n_features)
    tuning_curves = np.empty((n_classes, n_features))
    for i, loc in enumerate(feature_tuning_locs):
        curve = stats.vonmises.pdf(classes_rad, kappa=5, loc=loc)
        curve = curve_max * (curve / np.max(curve))
        tuning_curves[:, i] = curve

    y = classes.repeat(examples_per_class)

    # Sampling strategy: Use Sklar's theorem to sample correlated Poisson RV using
    # copula + Poisson marginals. We generate the copula using a multivariate normal.
    sigma = stats.invwishart.rvs(df=n_features, scale=np.eye(n_features))
    normal_samples = stats.multivariate_normal.rvs(cov=sigma, size=y.size)
    copula = stats.norm.cdf(normal_samples, scale=np.sqrt(np.diagonal(sigma)))
    X = stats.poisson.ppf(copula, mu=tuning_curves[y, :]).astype(int)

    return X, y
