"""Linear decoders for angled grating stimuli."""
from jax.config import config

from gdec.eld import EmpiricalLinearDecoder
from gdec.gppid import GPPoissonIndependentDecoder
from gdec.pid import PoissonIndependentDecoder
from gdec.snd import SuperNeuronDecoder

config.update("jax_enable_x64", True)


__all__ = [
    "EmpiricalLinearDecoder",
    "GPPoissonIndependentDecoder",
    "PoissonIndependentDecoder",
    "SuperNeuronDecoder",
]
