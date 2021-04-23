"""Linear decoders for angled grating stimuli."""
from jax.config import config

from gdec.eld import EmpiricalLinearDecoder
from gdec.gid import GaussianIndependentDecoder, LinearGaussianIndependentDecoder
from gdec.glmnet import LogisticRegression
from gdec.gpgid import GPGaussianIndependentDecoder
from gdec.gppid import GPPoissonIndependentDecoder
from gdec.pid import PoissonIndependentDecoder
from gdec.snd import SuperNeuronDecoder
from gdec.vgpmd import (
    VariationalGaussianProcessMulticlassDecoder as GaussianProcessMulticlassDecoder,
)

config.update("jax_enable_x64", True)

__all__ = [
    "EmpiricalLinearDecoder",
    "LinearGaussianIndependentDecoder",
    "GaussianIndependentDecoder",
    "GPGaussianIndependentDecoder",
    "GPPoissonIndependentDecoder",
    "LogisticRegression",
    "PoissonIndependentDecoder",
    "SuperNeuronDecoder",
    "VariationalGaussianProcessMulticlassDecoder",
    "GaussianProcessMulticlassDecoder",
]
