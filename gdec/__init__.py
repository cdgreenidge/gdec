"""Linear decoders for angled grating stimuli."""
from gdec.eld import EmpiricalLinearDecoder
from gdec.pid import PoissonIndependentDecoder
from gdec.snd import SuperNeuronDecoder

__all__ = ["EmpiricalLinearDecoder", "PoissonIndependentDecoder", "SuperNeuronDecoder"]
