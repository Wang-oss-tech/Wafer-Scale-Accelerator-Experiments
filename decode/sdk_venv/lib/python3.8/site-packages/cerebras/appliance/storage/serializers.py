# Copyright 2016-2025 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

import os
from functools import cached_property
from io import BytesIO
from typing import Optional, Union

import dill
import numpy
import numpy.lib.mixins

from cerebras.appliance.data.dtypes import bf16, is_bf16
from cerebras.appliance.storage import StorageReader, register_serializer
from cerebras.appliance.utils.classes import retrieve_all_subclasses


@register_serializer(bool, int, float)
class ScalarSerializer:
    """
    Class for serializing/deserializing python numeric.
    """

    __key__ = "scalar_value"

    @staticmethod
    def serialize(scalar, context):
        context.metadata[ScalarSerializer.__key__] = scalar

    @staticmethod
    def deserialize(array, context):
        return context.metadata[ScalarSerializer.__key__]


@register_serializer(str)
class StringSerializer:
    """
    Class for saving and loading python strings to and from the H5 checkpoint.
    """

    __key__ = "string_value"

    @staticmethod
    def serialize(s: str, context):
        context.metadata[StringSerializer.__key__] = s

    @staticmethod
    def deserialize(array, context):
        return context.metadata[StringSerializer.__key__]


@register_serializer(type(None))
class NoneSerializer:
    """
    Class for serializing/deserializing python Nones.
    """

    @staticmethod
    def serialize(s: str, context):
        pass

    @staticmethod
    def deserialize(obj, context):
        return None


@register_serializer(type)
class ObjectSerializer:
    """
    Fallback class for serializing/deserializing arbitrary python objects.
    """

    @staticmethod
    def serialize(obj, context):
        return BytesIO(dill.dumps(obj))

    @staticmethod
    def deserialize(b: BytesIO, context):
        return dill.loads(b.getbuffer())


@register_serializer()
class DeferredObject:
    def __init__(
        self,
        reader_or_path: Union[StorageReader, str],
        key: str,
        index: int = 0,
        metadata: Optional[dict] = None,
        force_external_link: Optional[bool] = None,
    ):
        self._reader_or_path = reader_or_path
        self._index = index
        self._metadata = metadata or {}
        self._force_external_link = force_external_link

        self._key = self._metadata.get("key", key)

        _ = self._reader  # Ensure reader is initialized

    @cached_property
    def _reader(self):
        if isinstance(self._reader_or_path, str):
            return StorageReader.get(self._reader_or_path)

        return self._reader_or_path

    @cached_property
    def _path(self):
        if isinstance(self._reader_or_path, str):
            return self._reader_or_path

        return self._reader_or_path.path

    @property
    def value(self):
        from .context import DeserializationContext

        obj, metadata = self._reader.read(self._key)
        metadata = {**self._metadata, **metadata}
        return DeserializationContext(self._key, metadata).deserialize(obj)

    # Override some dunders to make the deferred object behave like the
    # underlying object.
    # Note, this is not an exhaustive list of dunder methods. We can add more
    # as needed.
    def __getattr__(self, name):
        if not name.startswith("_") and name != "value":
            return getattr(self.value, name)
        return self.__getattribute__(name)

    def __getitem__(self, key):
        return self.value[key]

    def __call__(self, *args, **kwargs):
        return self.value(*args, **kwargs)

    def __iter__(self):
        return iter(self.value)

    def __len__(self):
        return len(self.value)

    def __contains__(self, item):
        return item in self.value

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.value})"

    def __getstate__(self):
        return {
            "path": str(self._path),
            "key": self._key,
            "index": self._index,
            "metadata": self._metadata,
        }

    def __setstate__(self, state):
        self.__init__(
            state["path"], state["key"], state["index"], state["metadata"]
        )

    @staticmethod
    def get(name: str):
        if name == "DeferredObject":
            return DeferredObject

        for subcls in retrieve_all_subclasses(DeferredObject):
            if subcls.__name__ == name:
                return subcls

        raise ValueError(f"Unknown deferred object class {name}")

    def serialize(self, context):
        same_file = os.path.realpath(context.ckpt_path) == os.path.realpath(
            self._reader.path
        )

        use_external_link = self._force_external_link or (
            not same_file and bool(context.use_external_link)
        )

        if not use_external_link:
            # When external links are disabled, we need to materialize the
            # tensor and save it to file. But note that we don't cache the
            # materialized tensor to avoid OOM.
            return context.hoist(context.serialize(self.value))

        context.metadata["deferred_object_cls"] = self.__class__.__name__
        context.metadata["path"] = str(self._reader.path)
        context.metadata["key"] = self._key
        context.metadata["index"] = self._index
        context.metadata["metadata"] = self._metadata

    def deserialize(value, context):
        if value is not None:  # Must already have been deserialized
            return value

        deferred_object_cls = DeferredObject.get(
            context.metadata["deferred_object_cls"]
        )
        return deferred_object_cls(
            context.metadata["path"],
            context.metadata["key"],
            context.metadata["index"],
            context.metadata["metadata"],
        )


@register_serializer()
class LinkedObject:
    def __init__(self, key: str):
        self.key = key

    def serialize(self, context):
        context.metadata["key"] = self.key
        context.metadata["__LINKED__"] = True

    def deserialize(value, context):
        return LinkedObject(context.metadata["key"])


@register_serializer(numpy.ndarray)
class NumpyArraySerializer:
    @staticmethod
    def serialize(array, context):
        context.metadata["is_bfloat16"] = is_bf16(array)
        return array

    @staticmethod
    def deserialize(array, context):
        if context.metadata.get("is_bfloat16"):
            return array.astype(bf16)
        return array


# Note, inheriting from NDArrayOperatorsMixin and overriding __array__ and
# __array_function_ is necessary to make the deferred numpy array behave like a
# numpy array.
@register_serializer()
class DeferredNumpyArray(
    DeferredObject, numpy.lib.mixins.NDArrayOperatorsMixin
):
    def __array__(self, dtype=None, copy=None):
        """Convert the deferred numpy array to a numpy array."""
        array = self.value
        if dtype is not None:
            array = array.astype(dtype)
        return array

    @staticmethod
    def recursive_value(value):
        if isinstance(value, DeferredNumpyArray):
            return value.value
        elif isinstance(value, dict):
            return type(value)(
                (k, DeferredNumpyArray.recursive_value(v))
                for k, v in value.items()
            )
        elif isinstance(value, (list, tuple)):
            return type(value)(map(DeferredNumpyArray.recursive_value, value))
        return value

    def __array_function__(self, func, types, args, kwargs):
        """Call numpy functions on the deferred numpy array."""
        return func(
            *DeferredNumpyArray.recursive_value(args),
            **DeferredNumpyArray.recursive_value(kwargs),
        )
