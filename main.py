from typing import Tuple
from jax.interpreters.xla import DeviceArray
from decorator import differentiable, Differentiable
from dataclasses import dataclass, field
from module import Module
from nn import Linear
from jax.nn import relu, log_softmax
import logging
from jax import numpy as np, random
import jax
from dataset import get_datasets, NumpyLoader

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


@jax.jit
def loss_fn(model: MLP, x: DeviceArray, y: DeviceArray) -> Tuple[DeviceArray, Tuple[MLP, DeviceArray]]:
    output = model(x)
    loss = cross_entropy(output, y)
    acc = accuracy(output, y)
    return loss, (model, acc)


@jax.jit
def sgd(param: DeviceArray, update: DeviceArray) -> DeviceArray:
    return param - 0.01 * update


@jax.jit
def accuracy(outputs: DeviceArray, targets: DeviceArray) -> DeviceArray:
    pred = np.argmax(outputs, axis=1)
    return np.mean(pred == targets)


def _one_hot(x, k, dtype=np.float32):
    """Create a one-hot encoding of x of size k."""
    return np.array(x[:, None] == np.arange(k), dtype)


def cross_entropy(outputs: DeviceArray, targets: DeviceArray) -> DeviceArray:
    probs = log_softmax(outputs)
    labels = _one_hot(targets, 10)
    loss = -np.mean(probs * labels)
    return loss


def main():
    logging.basicConfig(level=logging.INFO)

    in_features = 28 * 28
    hidden_dim = 1024
    out_features = 10
    batch_size = 128
    num_epochs = 10

    mlp = MLP.new(in_features, hidden_dim, out_features)
    # mlp = Linear.new(in_features, hidden_dim)
    rng = mlp.initialize()

    dloss_fn = jax.value_and_grad(loss_fn, has_aux=True)
    dloss_fn = jax.jit(dloss_fn)

    train_ds, val_ds = get_datasets()
    train_dl = NumpyLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
    )
    val_dl = NumpyLoader(
        val_ds,
        batch_size=batch_size
    )

    for epoch in range(num_epochs):

        for x, y in train_dl:
            x, y = np.array(x), np.array(y)

            (loss, (mlp, acc)), grads = dloss_fn(mlp, x, y)
            # # print('grad:', grad)
            print('train loss:', loss)
            print('train acc:', acc)

            mlp = jax.tree_multimap(sgd, mlp, grads)

        for x, y in val_dl:
            x, y = np.array(x), np.array(y)
            # out = mlp(x)

            loss, (mlp, acc) = loss_fn(mlp, x, y)

            print('val loss:', loss)
            print('val acc:', acc)

            


if __name__ == "__main__":
    main()
