"""Linear decoders for angled grating stimuli."""
from gdec.eld import EmpiricalLinearDecoder
from gdec.gppid import GPPoissonIndependentDecoder
from gdec.gpgid import GPGaussianIndependentDecoder
from gdec.pid import PoissonIndependentDecoder
from gdec.snd import SuperNeuronDecoder
from gdec.vgpmd import VariationalGaussianProcessMulticlassDecoder

__all__ = [
    "EmpiricalLinearDecoder",
    "GPGaussianIndependentDecoder",
    "GPPoissonIndependentDecoder",
    "PoissonIndependentDecoder",
    "SuperNeuronDecoder",
    "VariationalGaussianProcessMulticlassDecoder",
]
