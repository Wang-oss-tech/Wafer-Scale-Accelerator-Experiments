# Copyright 2016-2025 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

import logging
from io import BytesIO
from typing import Any, Protocol, Union, runtime_checkable

import numpy

from cerebras.appliance.storage import (
    DeserializationContext,
    SerializationContext,
)


@runtime_checkable
class Serializer(Protocol):
    @staticmethod
    def serialize(
        value: Any,
        context: SerializationContext,
    ) -> Union[numpy.ndarray, BytesIO, None]:
        """Serialize an object.

        To add to context metadata
        ```
        context.metadata["key"] = val
        ```

        To recursively serialize an object
        ```
        return context.serialize(other_object)
        ```

        Args:
            value: Object to serialize
            context: Serialization context

        Returns:
            This method must return either a numpy array, BytesIO, or None
        """

    @staticmethod
    def deserialize(object: Any, context: DeserializationContext):
        """Deserialize an object.

        To retrieve data from the context metadata
        ```
        val = context.metadata["key"]
        ```

        Args:
            object: The object to deserialize.
                If the original object was recursively serialized, this
                method will be called with the deserialized object.
                This means that no recursive deserialization should occur
                inside this method
            context: Deserialization context

        Returns:
            This method should return the fully deserialized object
        """


_SERIALIZER_REGISTRY = {}
_SERIALIZER_MAP = {}


def register_serializer(*types):
    def wrapper(serializer):
        assert isinstance(serializer, Serializer)
        for t in types or [serializer]:
            if t in _SERIALIZER_REGISTRY:
                if _SERIALIZER_REGISTRY[t] is not serializer:
                    raise ValueError(
                        f"Serializer already registered for type {t}. "
                        f"{_SERIALIZER_REGISTRY[t].__name__}. "
                        f"Cannot register multiple serializers for type {t}."
                    )
            else:
                _SERIALIZER_REGISTRY[t] = serializer

        if serializer.__name__ in _SERIALIZER_MAP:
            if _SERIALIZER_MAP[serializer.__name__] is not serializer:
                raise ValueError(
                    f"Serializer {serializer.__name__} already registered. "
                )
        else:
            _SERIALIZER_MAP[serializer.__name__] = serializer

        return serializer

    return wrapper


def get_serializer_for_value(value):
    tv = type(value)
    serializer = _SERIALIZER_REGISTRY.get(tv)
    if serializer is None:
        # Check __mro__ in case of inheritance
        for t in getattr(tv, "__mro__", ()):
            serializer = _SERIALIZER_REGISTRY.get(t)
            if serializer is not None:
                break
        else:
            logging.debug(
                f"No serializer registered for type {tv}. "
                f"Falling back to generic python object serialization"
            )
            serializer = _SERIALIZER_REGISTRY[type]

    return serializer


def get_serializer(name: str):
    if name not in _SERIALIZER_MAP:
        raise ValueError(f"No serializer registered with name {name}.")
    return _SERIALIZER_MAP[name]
