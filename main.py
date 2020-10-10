from typing import Tuple
from jax.interpreters.xla import DeviceArray
from decorator import differentiable, Differentiable
from dataclasses import dataclass, field
from module import Module
from nn import Linear
from jax.nn import relu
import logging
from jax import numpy as np, random
import jax

logger = logging.getLogger(__name__)


@differentiable
@dataclass
class MLP(Module):

    linear1: Linear = field(metadata=Differentiable)
    linear2: Linear = field(metadata=Differentiable)

    @jax.jit
    def __call__(self, x: DeviceArray) -> DeviceArray:
        x = self.linear1(x)
        x = relu(x)
        x = self.linear2(x)
        return x

    @classmethod
    def new(cls, in_features: int, hidden_dim: int, out_features: int):
        linear1 = Linear.new(in_features, hidden_dim)
        linear2 = Linear.new(hidden_dim, out_features)
        return cls(
            linear1=linear1,
            linear2=linear2
        )

# @jax.jit
def loss_fn(model: MLP, x: DeviceArray) -> Tuple[DeviceArray, MLP]:
    y = model(x)
    loss = y.sum()
    return loss, model

@jax.jit
def sgd(param: DeviceArray, update: DeviceArray) -> DeviceArray:
    return param - 0.01 * update


def main():
    logging.basicConfig(level=logging.INFO)

    in_features = 16
    hidden_dim = 64
    out_features = 32
    batch_size = 4

    mlp = MLP.new(in_features, hidden_dim, out_features)
    # mlp = Linear.new(in_features, hidden_dim)
    rng = mlp.initialize()

    x = random.normal(rng, [batch_size, in_features])

    dloss_fn = jax.value_and_grad(loss_fn, has_aux=True)
    dloss_fn = jax.jit(dloss_fn)

    for i in range(10):
        (loss, mlp), grads = dloss_fn(mlp, x)
        # print('grad:', grad)
        print('loss:', loss)

        mlp = jax.tree_multimap(sgd, mlp, grads)


if __name__ == "__main__":
    main()
