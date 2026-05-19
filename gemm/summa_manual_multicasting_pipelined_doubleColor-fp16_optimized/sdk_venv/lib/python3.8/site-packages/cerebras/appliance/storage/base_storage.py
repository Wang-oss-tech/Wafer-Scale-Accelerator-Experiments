# Copyright 2016-2025 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from io import BytesIO
from typing import Optional, Tuple, Union

import numpy

from cerebras.appliance.utils.classes import retrieve_all_subclasses


class _BaseAbstractStorage(ABC):
    def __init__(self, path: str):
        self._path = path

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    @property
    def path(self):
        return self._path

    @classmethod
    def get_subclasses(cls):
        yield from retrieve_all_subclasses(cls)

    @classmethod
    @abstractmethod
    def is_valid_path(cls, path: str) -> bool:
        """Returns true if there is any subclass that can handle the path."""
        return any(
            subcls.is_valid_path(path) for subcls in cls.get_subclasses()
        )

    @classmethod
    @abstractmethod
    def path_exists(cls, path: str) -> bool:
        """Returns true if the path exists."""
        return any(
            subcls.path_exists(path)
            for subcls in cls.get_subclasses()
            if subcls.is_valid_path(path)
        )

    @classmethod
    def construct(cls, path, *args, **kwargs):
        return cls(path, *args, **kwargs)

    @classmethod
    def get_type(cls, path: str):
        path = str(path)
        matches = [
            subcls
            for subcls in cls.get_subclasses()
            if subcls.is_valid_path(path)
        ]
        if len(matches) == 0:
            raise ValueError(
                f"Could not find {cls.__name__} for invalid path: {path}"
            )
        if len(matches) > 1:
            raise ValueError(
                f"Multiple {cls.__name__}s found for path: {path}.\n"
                f"{sorted(subcls.__name__ for subcls in matches)}\n"
                f"Please ensure that all {cls.__name__} are unique."
            )

        return matches[0]

    @classmethod
    def get(cls, path: str, *args, **kwargs):
        return cls.get_type(path).construct(path, *args, **kwargs)


class StorageReader(_BaseAbstractStorage):
    @dataclass
    class Stats:
        # Path to object the following stats correspond to
        path: str
        # Time of most recent content modification expressed in seconds.
        st_mtime: int

    def exists(self) -> bool:
        return self.path_exists(self.path)

    @property
    @abstractmethod
    def stats(self) -> "StorageReader.Stats":
        """Returns the stats of the object at the path."""

    @abstractmethod
    def read(self, key: Optional[str]) -> Tuple[BytesIO, dict]:
        """Reads the object at the key and returns the object and metadata.


        Args:
            key: The key to read from.
                If key is None, return global metadata.

        Returns:
            Tuple of object and metadata.
        """

    @cached_property
    def global_metadata(self):
        _, metadata = self.read(None)
        return metadata


class DeferredStorageReader(StorageReader):
    def __init__(self, path: str):
        super().__init__(path)

    @classmethod
    def is_valid_path(cls, path: str) -> bool:
        # This is a deferred reader, so it doesn't directly
        # handle any path
        return False

    @classmethod
    def path_exists(cls, path: str) -> bool:
        # This is a deferred reader, so it doesn't directly
        # handle any path
        return False

    @cached_property
    def reader(self):
        return StorageReader.get(self.path)

    class DeferredStats(StorageReader.Stats):
        def __init__(self, deferred_reader):
            self.deferred_reader = deferred_reader

        @property
        def path(self):
            return self.deferred_reader.path

        @cached_property
        def st_mtime(self):
            return self.deferred_reader.reader.stats.st_mtime

    @property
    def stats(self):
        return DeferredStorageReader.DeferredStats(self)

    def read(self, key: Optional[str]) -> Tuple[BytesIO, dict]:
        return self.reader.read(key)


class StorageWriter(_BaseAbstractStorage):
    def write(
        self,
        key: str,
        object: Union[BytesIO, numpy.ndarray, None] = None,
        metadata: dict = None,
    ):
        metadata = metadata or {}
        if isinstance(object, numpy.ndarray):
            self.write_numpy(key, object, metadata)
        elif isinstance(object, BytesIO):
            self.write_bytes(key, object, metadata)
        elif object is None:
            self.write_metadata(key, metadata)
        else:
            raise Exception(type(object))

    @abstractmethod
    def write_numpy(self, key: str, array: numpy.ndarray, metadata: dict): ...

    @abstractmethod
    def write_bytes(self, key: str, bytes: BytesIO, metadata: dict): ...

    @abstractmethod
    def write_metadata(self, key: str, metadata: dict):
        # write metadata at key
        # If key is None, write global metadata
        ...

    @classmethod
    def get(cls, path: str, *args, **kwargs):
        writer = super().get(path, *args, **kwargs)
        if writer.path_exists(path):
            raise FileExistsError(
                f"Checkpoint at '{path}' already exists. "
                "Please delete the checkpoint and run again."
            )
        return writer


class StorageDeleter(_BaseAbstractStorage):
    @abstractmethod
    def delete(self, key: Optional[str] = None) -> bool:
        """Delete the object at the key.

        If no key is specified, delete the whole checkpoint

        Args:
            key: The key to delete. If None, delete the whole checkpoint.

        Returns:
            True if the object/checkpoint was deleted, False otherwise.
        """
