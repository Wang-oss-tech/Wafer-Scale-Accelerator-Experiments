# Copyright 2016-2025 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

import json
from collections import namedtuple
from contextlib import contextmanager
from dataclasses import dataclass
from io import BytesIO
from typing import Any

import numpy

__TYPE__ = "__TYPE__"


@dataclass
class SerializedObject:
    object: Any
    metadata: dict


# Specifically use namedtuple instead of dataclass
# to support easier type serialization
NestedObject = namedtuple("NestedObject", ["object", "metadata"])


@dataclass
class SerializedMetadata:
    metadata: dict


@dataclass
class HoistedObject:
    object: Any


class SerializationContext:
    def __init__(
        self, ckpt_path: str, key: str, use_external_link: bool = False
    ):
        self.ckpt_path = ckpt_path
        self.key = key
        self.use_external_link = use_external_link

        self.metadata_stack = []
        # Keep track of objects we've seen
        # to prevent infinite recursion
        self.seen = set()

    @property
    def metadata(self):
        return self.metadata_stack[-1]

    @contextmanager
    def serialization_context(self, value):
        if id(value) in self.seen:
            raise Exception("Infinite Recursion!")

        self.metadata_stack.append({})
        self.seen.add(id(value))
        try:
            yield
        finally:
            self.metadata_stack.pop()
            self.seen.remove(id(value))

    def _set_special_metadata(self, key, value):
        if key in self.metadata:
            raise KeyError(f"{key} is a special key and must not be set")

        self.metadata[key] = value

    def serialize(self, value):
        from cerebras.appliance.storage import get_serializer_for_value

        with self.serialization_context(value):
            serializer = get_serializer_for_value(value)
            serialized = serializer.serialize(value, self)

            self._set_special_metadata(__TYPE__, serializer.__name__)

            try:
                json.dumps(self.metadata)
            except TypeError as e:
                raise TypeError(f"Metadata must be JSON serializable") from e

            if serialized is None:
                self._set_special_metadata("__NONE__", True)
                return SerializedObject(serialized, self.metadata)

            elif isinstance(serialized, numpy.ndarray):
                self._set_special_metadata("__NUMPY__", True)
                return SerializedObject(serialized, self.metadata)

            elif isinstance(serialized, BytesIO):
                self._set_special_metadata("__BYTES__", True)
                return SerializedObject(serialized, self.metadata)

            elif isinstance(serialized, HoistedObject):
                # Ignore the current level of metadata, and return
                # the object as is (i.e. hoist it up)
                return serialized

            else:
                return NestedObject(
                    serialized, SerializedMetadata(self.metadata)
                )

    def hoist(self, value):
        """Hoist the object up one level.

        This is useful when we want to serialize an object
        but not include the metadata at the current level.
        """
        if not isinstance(value, (SerializedObject, NestedObject)):
            raise ValueError(f"Cannot hoist non-serialized object: {value}")
        return HoistedObject(value)


class DeserializationContext:
    def __init__(self, key, metadata):
        self.key = key
        self.metadata = metadata

    def deserialize(self, value):
        from cerebras.appliance.storage import get_serializer

        if __TYPE__ not in self.metadata:
            raise ValueError(
                f"Missing {__TYPE__} in metadata for {self.key}: {self.metadata}"
            )

        serializer = get_serializer(self.metadata[__TYPE__])
        return serializer.deserialize(value, self)
