from jax.interpreters.xla import DeviceArray
from jax.tree_util import register_pytree_node
from typing import Any, Callable, ClassVar, Dict, List, NewType, Optional, Tuple, Type, TypeVar, Union
from dataclasses import Field, dataclass
from jax import numpy as np
from module import Module
import itertools
import logging


logger = logging.getLogger(__name__)

# Parameter = NewType('Parameter', DeviceArray)
# Buffer = NewType('Buffer', DeviceArray)

_DIFFERENTIABLE = 'differentiable'
_NON_DIFFERENTIABLE = 'non_differentiable'
_FIELDS = '__dataclass_fields__'

Differentiable = {
    _DIFFERENTIABLE: True
}
NonDifferentiable = {
    _DIFFERENTIABLE: False
}


T = TypeVar('T')


def _get_keys(node: Any) -> Dict[str, List[str]]:
    keys: Dict[str, List[str]] = {
        'differentiable': [],
        'non_differentiable': [],
    }

    fields: Dict[str, Field] = getattr(node, _FIELDS)

    for key, value in fields.items():
        if value.metadata.get(_DIFFERENTIABLE, False):
            keys[_DIFFERENTIABLE].append(key)
        else:
            keys[_NON_DIFFERENTIABLE].append(key)

    return keys


def differentiable(cls: Type[T]) -> Type[T]:

    keys = _get_keys(cls)

    def _tree_flatten(node: Module) -> Tuple[Tuple[Dict[str, Any]], Dict[str, Any]]:
        children = {}
        aux_data = {}
        for key in keys[_DIFFERENTIABLE]:
            children[key] = getattr(node, key)

        for key in keys[_NON_DIFFERENTIABLE]:
            aux_data[key] = getattr(node, key)

        logger.debug('=' * 50)
        logger.debug('flatten: %s', cls)
        logger.debug('aux_data: %s', aux_data)
        logger.debug('children: %s', children)
        return (children,), aux_data

    def _tree_unflatten(aux_data: Tuple[Dict[str, Any]], children: Dict[str, Any]) -> Module:
        logger.debug('=' * 50)
        logger.debug('unflatten: %s', cls)
        logger.debug('aux_data: %s', aux_data)
        logger.debug('children: %s', children)
        return cls(**aux_data, **children[0])  # type: ignore

    register_pytree_node(cls, _tree_flatten, _tree_unflatten)

    return cls
