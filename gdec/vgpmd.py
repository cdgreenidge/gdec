"""Variational Gaussian Process multiclass Decoder."""
import logging
import math
from typing import Any, Tuple

import numpy as np
import sklearn.base
import sklearn.utils.multiclass
import sklearn.utils.validation
import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from gdec import torchgp, useful


def logit(x: float) -> float:
    """Logit function."""
    return math.log(x / (1 - x))


def interval_to_real(x: float, low: float, high: float) -> float:
    return logit((x - low) / (high - low))


def real_to_interval(x: torch.Tensor, low: float, high: float) -> torch.Tensor:
    return (high - low) * torch.sigmoid(x) + low


def prune(spectrum: torch.Tensor) -> torch.Tensor:
    """Prunes the spectrum and its associated matrices."""
    condition_thresh = 1e16
    spectrum_thresh = torch.max(torch.abs(spectrum ** 2)) / condition_thresh
    mask = torch.nonzero(
        (torch.abs(spectrum ** 2) <= spectrum_thresh)
        & (torch.abs(spectrum ** 2) <= 1e-64),
        as_tuple=True,
    )
    spectrum[mask] = 0
    return spectrum


class VGPMDModule(nn.Module):
    """Variational Gaussian Process Linear Decoder."""

    def __init__(self, n_dim: int, n_classes: int, n_samples: int = 3) -> None:
        super().__init__()
        self.register_buffer("n_dim", torch.tensor(n_dim))
        self.register_buffer("n_classes", torch.tensor(n_classes))
        self.register_buffer("n_samples", torch.tensor(n_samples))
        self.log_amplitudes = nn.Parameter(
            torch.full((n_dim,), math.log(0.01)), requires_grad=True
        )
        self.unconstrained_lengthscales = nn.Parameter(
            torch.full(
                (n_dim,), interval_to_real(0.055 * n_classes, low=0.01, high=n_classes)
            ),
            requires_grad=True,
        )

        # Variational posterior parameters, for torchgp-domain weights
        self.q_mean = nn.Parameter(torch.empty((n_classes, n_dim)), requires_grad=True)
        nn.init.xavier_normal_(self.q_mean)
        self.log_q_scale = nn.Parameter(
            torch.full((n_classes, n_dim), math.log(1.0e-5)), requires_grad=True
        )

        basis, freqs = torchgp.real_fourier_basis(n_classes)
        self.register_buffer("basis", basis)
        self.register_buffer("freqs", freqs)

    def amplitudes(self) -> torch.Tensor:
        """Return the constrained amplitudes."""
        return torch.exp(self.log_amplitudes)

    def lengthscales(self) -> torch.Tensor:
        """Return the constrained lengthscales."""
        return real_to_interval(
            self.unconstrained_lengthscales,
            low=0.01,
            high=float(self.n_classes.item()),  # type: ignore
        )

    def kl_penalty(self) -> torch.Tensor:
        """Compute the KL divergence prior penalty, used for constructing the ELBO."""
        q = dist.Independent(
            dist.Normal(self.q_mean, torch.exp(self.log_q_scale)),
            reinterpreted_batch_ndims=2,
        )
        p_mean = torch.zeros_like(self.q_mean)
        p_scale = torch.ones_like(self.q_mean)
        p = dist.Independent(dist.Normal(p_mean, p_scale), reinterpreted_batch_ndims=2)
        return dist.kl_divergence(q, p)

    def coefs(self) -> torch.Tensor:
        """Return a MAP estimate of the unwhitened coefficients."""
        spectrum = torchgp.rbf_spectrum(
            self.freqs, self.amplitudes(), self.lengthscales()  # type: ignore
        ).t()
        return torch.sqrt(spectrum) * self.q_mean

    def forward(  # type: ignore
        self, X: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the class log-probabilities for X.

        Args:
            X: The input minibatch, of shape (n_data, n_dim).

        Returns:
            A tuple of the log probabilities and the KL divergence penalty.In
            train mode, the log probabilities are mean of a batch of class log
            probabilities with weights sampled from the posterior. (computing
            the means now avoids an expensive gather op later). In eval mode,
            we sample the weights from the posterior mean. The log
            probabilities are of shape (n_data, n_classes). The KL divergence
            penalty is a scalar, and it is not computed (so set to None) when
            in eval mode.

        """
        assert X.size()[1] == self.n_dim
        q = dist.Normal(self.q_mean, torch.exp(self.log_q_scale))
        spectrum = torchgp.rbf_spectrum(
            self.freqs, self.amplitudes(), self.lengthscales()  # type: ignore
        ).t()
        spectrum = prune(spectrum)
        if self.training:  # type: ignore
            U = q.rsample((self.n_samples,))
            weights = self.basis @ (torch.sqrt(spectrum) * U)
            log_probs = torch.mean(
                F.log_softmax(X @ torch.transpose(weights, dim0=-2, dim1=-1), dim=-1),
                dim=0,
            )
        else:
            U = self.q_mean
            weights = self.basis @ (torch.sqrt(spectrum) * U)
            log_probs = F.log_softmax(X @ weights.t(), dim=-1)
        kl_penalty = self.kl_penalty()  # We could skip this for speed later
        return log_probs, kl_penalty


class NegativeELBO(nn.Module):
    """Negative ELBO loss.

    Args:
        n_data: The total number of examples in the training dataset.

    """

    def __init__(self, n_data: int) -> None:
        super().__init__()
        self.register_buffer("n_data", torch.tensor(n_data))

    def forward(  # type: ignore
        self, y_pred: Tuple[torch.Tensor, torch.Tensor], y_target: torch.Tensor
    ) -> torch.Tensor:
        """Compute the negative ELBO.

        Args:
            y_pred: The output of the VGPMD decoder.
            y_target: The true class labels, zero-based.

        Returns:
            The negative ELBO.

        """
        log_probs, kl_penalty = y_pred
        likelihood = self.n_data * -F.nll_loss(log_probs, y_target, reduction="mean")
        return -likelihood + kl_penalty


def fit_tuning_curve_matrix(
    X: np.ndarray,
    y: np.ndarray,
    lr: float = 0.1,
    max_steps: int = 4096,
    n_samples: int = 4,
    log_every: int = 32,
    cuda: bool = True,
    cuda_device: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if torch.cuda.is_available() and cuda:
        device = torch.device("cuda", cuda_device)
    else:
        device = torch.device("cpu")

    n_classes = np.unique(y).size
    X = torch.tensor(X).to(device)
    y = torch.tensor(y).to(device)
    model = VGPMDModule(n_dim=X.shape[1], n_classes=n_classes, n_samples=n_samples).to(
        device
    )
    loss_fn = NegativeELBO(n_data=X.shape[0]).to(device)
    optimizer = optim.Adam(
        (
            {
                "params": (model.log_amplitudes, model.unconstrained_lengthscales),
                "lr": 1.0 * lr,
            },
            {"params": (model.q_mean, model.log_q_scale)},
        ),
        lr=lr,
        amsgrad=True,
    )
    patience = 32
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=patience, verbose=True, cooldown=256
    )

    for step in range(max_steps):
        optimizer.zero_grad()
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()

        if step > 512:
            scheduler.step(loss)
            last_lr = torch.tensor(scheduler._last_lr)  # type: ignore
            eps = scheduler.eps  # type: ignore
            if (last_lr <= eps).all():
                break

        if step % log_every == 0:
            logging.info("Step %d, ELBO %.2f", step, loss)

    with torch.no_grad():
        freq_coefs = model.coefs()
        coefs = model.basis @ freq_coefs  # type: ignore
        amplitudes = model.amplitudes()
        lengthscales = model.lengthscales()

    return freq_coefs.cpu(), coefs.cpu(), amplitudes.cpu(), lengthscales.cpu()


class VariationalGaussianProcessMulticlassDecoder(
    sklearn.base.BaseEstimator, sklearn.base.ClassifierMixin
):
    """Sklearn interface to VGPMD classifier.

    Args:
        affine: whether or not to include a linear intercept term.

    """

    def __init__(self, *args: Any, affine: bool = False, **kwargs: Any) -> None:
        self.affine = affine
        super().__init__(*args, **kwargs)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        lr: float = 0.1,
        max_steps: int = 4096,
        n_samples: int = 4,
        log_every: int = 32,
        cuda: bool = True,
        cuda_device: int = 0,
    ) -> "VariationalGaussianProcessMulticlassDecoder":
        """Fit the estimator."""
        X, y = sklearn.utils.validation.check_X_y(X, y)
        if self.affine:
            X = useful.add_intercept_feature_col(X)

        self.classes_ = sklearn.utils.multiclass.unique_labels(y)
        self.X_ = X.astype(np.float32)
        self.y_ = y
        # t suffix stands for torch
        (
            self.freq_coefs_t_,
            self.coefs_t_,
            amplitudes,
            lengthscales,
        ) = fit_tuning_curve_matrix(
            self.X_, self.y_, lr, max_steps, n_samples, log_every, cuda, cuda_device,
        )
        self.amplitudes_ = amplitudes.numpy()
        self.lengthscales_ = lengthscales.numpy()

        return self

    @property
    def coefs_(self) -> np.ndarray:
        """Return the model coefficients, of size (k, d)."""
        sklearn.utils.validation.check_is_fitted(self)
        return self.coefs_t_.numpy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict classes."""
        sklearn.utils.validation.check_is_fitted(self)
        X = sklearn.utils.validation.check_array(X)
        if self.affine:
            X = useful.add_intercept_feature_col(X)
        X = torch.tensor(X, dtype=torch.float32)
        log_probs = F.log_softmax(X @ self.coefs_t_.t(), dim=-1)
        classes = torch.argmax(log_probs, dim=-1)
        return classes.numpy()

    def resample(self, n_classes: int) -> "VariationalGaussianProcessMulticlassDecoder":
        """Resample model to a different number of classes."""
        basis = torchgp.real_fourier_basis(n_classes)[0]
        new_coefs_t_ = basis @ self.freq_coefs_t_

        model = VariationalGaussianProcessMulticlassDecoder()
        model.classes_ = np.arange(n_classes)
        model.X_ = self.X_
        model.y_ = self.y_
        model.coefs_t_ = new_coefs_t_
        model.freq_coefs_t_ = self.freq_coefs_t_
        return model
