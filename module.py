from dataclasses import dataclass
from typing import Optional

from jax.interpreters.xla import DeviceArray
from jax import random


class Module:

    def _reset_parameters(self, rng: DeviceArray):
        pass

    def initialize(self, seed: int = 42, recurse: bool = True, rng: Optional[DeviceArray] = None) -> DeviceArray:
        if rng is None:
            rng = random.PRNGKey(seed)

        self._reset_parameters(rng)

        if recurse:
            for _key, value in self.__dict__.items():
                if isinstance(value, Module):
                    rng, module_rng = random.split(rng)
                    value.initialize(
                        seed=seed, recurse=recurse, rng=module_rng
                    )

        return rng

    def _train(self, mode=True):
        pass

    def train(self, mode=True):
        self._train(mode=mode)

        for _key, value in self.__dict__.items():
            if isinstance(value, Module):
                value.train(mode=mode)

        return self
