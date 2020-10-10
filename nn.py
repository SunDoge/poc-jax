from typing import Optional
from jax.interpreters.xla import DeviceArray
from module import Module
from dataclasses import dataclass, field
from decorator import differentiable, Differentiable, NonDifferentiable
from jax import numpy as np, random
import logging
import jax

logger = logging.getLogger(__name__)


@differentiable
@dataclass
class Linear(Module):

    in_features: int
    out_features: int
    use_bias: bool

    weight: DeviceArray = field(metadata=Differentiable)
    bias: Optional[DeviceArray] = field(metadata=Differentiable)

    @jax.jit
    def __call__(self, input: DeviceArray) -> DeviceArray:
        output = np.dot(input, self.weight.T)

        if self.bias is not None:
            output += self.bias

        return output

    def _reset_parameters(self, rng: DeviceArray):
        k1, k2 = random.split(rng)

        self.weight = random.normal(k1, self.weight.shape)
        logger.debug('self.weight: %s', self.weight.shape)

        if self.bias is not None:
            self.bias = random.normal(k2, self.bias.shape)
            logger.debug('self.bias: %s', self.bias.shape)

    @classmethod
    def new(cls, in_features: int, out_features: int, use_bias: bool = True):
        weight = np.empty([out_features, in_features])
        if use_bias:
            bias = np.empty([out_features])
        else:
            bias = None

        return cls(
            in_features=in_features,
            out_features=out_features,
            use_bias=use_bias,
            weight=weight,
            bias=bias
        )
