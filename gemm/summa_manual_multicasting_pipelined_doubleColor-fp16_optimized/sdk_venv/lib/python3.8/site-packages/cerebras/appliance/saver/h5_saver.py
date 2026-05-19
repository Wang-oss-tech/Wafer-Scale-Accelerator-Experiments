# Copyright 2016-2023 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

"""
HDF5 based checkpoint saver.
"""

from __future__ import annotations

import enum
import os
from inspect import isclass
from typing import Any, Dict, List, Optional
from urllib.parse import quote, unquote  # to escape `/` in tensor names
from warnings import warn

import dill
import h5py as h5
import hdf5plugin
import numpy as np
from numpy import ndarray as nd

from cerebras.appliance.data.dtypes import is_bf16
from cerebras.appliance.log import ClassLogger, named_class_logger
from cerebras.appliance.saver.base_saver import BaseSaver
from cerebras.appliance.utils._contexts import ValueContext
from cerebras.appliance.utils.misc import is_cerebras_available

H5_TYPE_KEY = "__H5_TYPE__"


class Storage(str, enum.Enum):
    """Strategy to store tensors."""

    TENSOR = "COMPACT"  # store tensor along with metadata
    PROTOBUF = "EXTERNAL"  # storage is external file
    SHARDED = "VIRTUAL"  # multiple datasets are used to represent tensors.

    @staticmethod
    def get_storage_type(name: str):
        """Returns the StorageType enum for the given storage string."""
        # TODO this is probably redundant
        return Storage(name.upper())


class Compression(str, enum.Enum):
    """
    Compression to use for storage.
    """

    NONE = None
    GZIP = "gzip"
    SZIP = "szip"
    LZF = "lzf"
    LZ4 = "lz4"

    @staticmethod
    def get_compression(name: str):
        """Returns Compression Enum for the given string."""
        return Compression(name.lower())

    @classmethod
    def _missing_(cls, value):
        """Default to no compression."""
        return Compression.NONE

    @staticmethod
    def get_compression_options(compression: Compression):
        """Return compression options for the given compression type."""

        if compression == Compression.GZIP:
            return 0  # TODO make this configurable
        if compression == Compression.SZIP:
            return ''
        if compression == Compression.LZF:
            return None
        else:
            return ''


DEFAULT_NBYTES: int = 1024 * 1024  # 1 MB
DEFAULT_STORAGE_TYPE: Storage = Storage.TENSOR
DEFAULT_COMPRESSION: Compression = Compression.LZ4
hdf5_locking = ValueContext("best-effort")


@named_class_logger("H5Saver")
class H5Saver(BaseSaver, ClassLogger):
    """
    HDF5 format backed checkpointing class to save/restore numpy arrays and scalars.
    Checkpoint for each step is saved as a separate H5 file objects with each tensor
    in a separate hdf5 dataset allowing partial access of checkpoint files.
    """

    def __init__(
        self,
        nbytes: int = DEFAULT_NBYTES,
        storage_type: Storage = DEFAULT_STORAGE_TYPE,
        compression: Compression = DEFAULT_COMPRESSION,
    ):
        """
        Constructs a H5 Saver object.

        Args:
            nbytes: number of bytes to load into memory from hdf5 datasets, defaults to 1 MB
            storage_type: tensor storage strategy, defaults to Storage.TENSOR tensor
            compression: Compression strategy, defaults to Compression.GZIP
        """
        super().__init__()
        # these should only be set only when creating
        # except for nbytes, other config parameters cannot change
        self.nbytes = nbytes
        self.storage_type = storage_type
        self.compression = compression
        # Maps ckpt_path to f.keys()
        self._keys = {}

        if self.compression == Compression.NONE:
            self.compress_opts = {}
        elif self.compression == Compression.LZ4:
            self.compress_opts = hdf5plugin.LZ4()
        else:
            self.compress_opts = {
                "compression": self.compression,
                "compression_opts": Compression.get_compression_options(
                    self.compression
                ),
            }

    # pylint: disable=no-self-use
    def _create_ckpt_dirs(self, ckpt_path: str):
        """Create checkpoint directories if it does not exist."""
        parent_dir = os.path.dirname(ckpt_path)
        if parent_dir and not os.path.exists(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)

    def save_tensor(
        self, ckpt_file: str, tensor_name: str, tensor_value: nd
    ) -> None:
        """Saves a tensor value as a separate h5 dataset for the given file.

        Args:
            ckpt_file: where to write checkpoint
            tensor_name: name of the tensor, this will be used as the h5 dataset name
            tensor_value: numpy ndarray or numpy scalar
        """
        self.save(ckpt_file, {tensor_name: tensor_value})

    def load_tensor(self, ckpt_path: str, tensor_name: str) -> nd:
        """Loads the tensor with given name from the path provided.

        Args:
            ckpt_path: a specific checkpoint
            tensor_name: name of the tensor to get

        Returns:
            Tensor value
        """
        return self.load(ckpt_path, [tensor_name])[tensor_name]

    @staticmethod
    def tensor_names(ckpt_path: str) -> List[str]:
        """Returns all the tensor names for a given checkpoint.

        Args:
            ckpt_path: a specific checkpoint

        Returns:
            List of tensor names
        """
        tensor_names = []
        ckpt_file = ckpt_path
        if os.path.exists(ckpt_file):
            with h5.File(ckpt_file, 'r', locking=hdf5_locking.value) as f:
                tensor_names = [unquote(x) for x in f.keys()]
        return tensor_names

    def save(self, ckpt_file: str, tensor_dict: Dict[str, nd]) -> str:
        """Save tensor_dict to provided file.

        Args:
            ckpt_file: where to write checkpoint
            tensor_dict: tensor_name, numpy ndarray or numpy scalar pairs

        Returns:
            checkpoint file path
        """
        self._create_ckpt_dirs(ckpt_file)
        # Clear any potential f.keys() cache for file we're modifying.
        self._keys.pop(ckpt_file, None)
        with h5.File(ckpt_file, "a", locking=hdf5_locking.value) as f:
            for name, val in tensor_dict.items():
                self._save_tensor_to_checkpoint(
                    f, name, val, self.compress_opts
                )

        return ckpt_file

    def _save_tensor_to_checkpoint(
        self,
        f: h5.File,
        tensor_name: str,
        tensor: nd,
        compression_opts: Optional[dict] = None,
    ):
        compression_opts = compression_opts or self.compress_opts

        self.logger.debug(
            f"Saving tensor `{tensor_name}` with type `{type(tensor)}`"
        )
        h5_type = SUPPORTED_H5_TYPES.get(type(tensor))
        if h5_type is None:
            self.logger.debug(
                f"Unsupported type for {tensor_name}: {type(tensor)}. "
                f"Falling back to generic python object saver"
            )
            h5_type = SUPPORTED_H5_TYPES[type]

        try:
            tensor_name = quote(tensor_name, safe='')
            save_type = h5_type.save(tensor, f, tensor_name, **compression_opts)
            if save_type is None:
                save_type = h5_type
            else:
                assert isinstance(save_type, type)
            # Save class name that was used to save
            # Note: if f[tensor_name] is an external link, setting the attr
            # actually changes the attr on the target file instead of this file.
            f[tensor_name].attrs[H5_TYPE_KEY] = save_type.__name__

            if f[tensor_name].external is not None:
                warning_message = (
                    "Saving tensors using external files may cause undefined "
                    "behavior during loading."
                )
                if is_cerebras_available():
                    warning_message += " Please see SW-98658 for more details."

                warn(warning_message)

                for filepath, _offset, size in f[tensor_name].external:
                    if size > 2**30:
                        raise ValueError(
                            f"External size is too large for file {filepath}. "
                            f"This will cause issues when loading the value from "
                            f"the checkpoint. Please split the external file into "
                            f"smaller chunks."
                        )

        except Exception as e:
            raise RuntimeError(
                f"Failed to save {tensor_name} with type {type(tensor)}."
            ) from e

    def create_links(
        self, file_name: str, original_index: str, aliases: List[str]
    ):
        """Create soft link for duplicate entries with different names."""
        if aliases:
            # Clear any potential f.keys() cache for file we're modifying.
            self._keys.pop(file_name, None)
            with h5.File(file_name, "a", locking=hdf5_locking.value) as f:
                for index in aliases:
                    f[index] = f[original_index]

    def load(
        self, ckpt_file: str, tensor_names: Optional[List[str]] = None
    ) -> Dict[str, nd]:
        """Load all tensors for the given checkpoint.

        Args:
            ckpt_file: checkpoint to load
            tensor_names: which tensors to load (Defaults to all)

        Returns:
            Mapping from tensor_name to tensor values
        """
        assert os.path.exists(
            ckpt_file
        ), f"Could not find checkpoint: {ckpt_file}"
        tensor_dict = {}
        with h5.File(ckpt_file, "r", locking=hdf5_locking.value) as f:
            # Provided names are unquoted so unquote for consistency
            if ckpt_file in self._keys:
                # Cache f.keys(), as scanning the entire h5 file can be quite
                # slow if file locking was used.
                keys = self._keys[ckpt_file]
            else:
                keys = [unquote(name) for name in f.keys()]
                self._keys[ckpt_file] = keys
            if tensor_names is None:
                tensor_names = keys

            keys = set(keys)

            for name in tensor_names:
                if name not in keys:
                    raise KeyError(
                        f"Could not find key `{name}` in checkpoint {ckpt_file}"
                    )

                dset = f[quote(name, safe='')]

                if dset.external is not None:
                    for filepath, _offset, size in dset.external:
                        if size > 2**30:
                            error_message = (
                                f"External size is too large for file {filepath}. "
                                f"This will cause issues when loading the value from "
                                f"the checkpoint."
                            )
                            if is_cerebras_available():
                                error_message += (
                                    " Please see SW-98658 for more details."
                                )

                            raise ValueError(error_message)

                try:
                    tensor_dict[name] = self._load_tensor_from_checkpoint(
                        f, quote(name, safe='')
                    )
                except OSError as e:
                    if not dset.external:
                        raise
                    # If the dataset is external, raise a more helpful
                    # message by adding the external filepaths.
                    files = ", ".join([e[0] for e in dset.external])
                    raise OSError(
                        f"Failed to load tensor `{name}` from "
                        f"HDF5 file `{ckpt_file}`. This tensor's "
                        f"content is expected to be stored in the "
                        f"following external file(s): {files}. Please "
                        f"ensure that the external files are "
                        f"accessible and valid."
                    ) from e

        return tensor_dict

    def _load_tensor_from_checkpoint(self, f: h5.File, key: str) -> Any:
        dset = f[key]

        # Load the value using the saved H5Type class
        h5_type_name = dset.attrs.get(H5_TYPE_KEY, None)
        if h5_type_name is None:
            raise UnknownH5TypeError(f"Could not find H5Type for {key}")

        return self._load_by_typename(h5_type_name, f, key)

    def _load_by_typename(self, h5_type_name: str, f: h5.File, key: str) -> Any:
        """Load the value using the given H5Type class."""
        if h5_type_name not in H5_TYPES_MAP:
            raise KeyError(
                f"Found unsupported H5Type in checkpoint. "
                f"Expected one of {sorted(H5_TYPES_MAP)}. "
                f"Got {h5_type_name}."
            )

        return H5_TYPES_MAP[h5_type_name].load(f, key)

    @staticmethod
    def is_valid_checkpoint(file_path: Optional[str]) -> bool:
        """Check if file is correct format to be a checkpoint."""
        if file_path is None:
            return False
        return h5.is_hdf5(file_path)


SUPPORTED_H5_TYPES = {}
H5_TYPES_MAP = {}


def register_h5_type(*types):
    """Decorator to register H5Type classes to a list of python types."""

    def cls_wrapper(cls):
        if cls.__name__ not in H5_TYPES_MAP:
            H5_TYPES_MAP[cls.__name__] = cls
        elif H5_TYPES_MAP[cls.__name__] is not cls:
            raise RuntimeError(
                f"Cannot register H5Type {cls} as there already exists "
                f"an H5Type named {cls.__name__} registered to "
                f"{H5_TYPES_MAP[cls.__name__]}"
            )
        if not (hasattr(cls, "save") and hasattr(cls, "load")):
            raise TypeError(
                f"Expected H5Type to have static `save` and `load` methods. "
                f"Please implement them"
            )

        for t in types or [cls]:
            assert isclass(t), f"Failed to register H5Type for non-type: {t}"
            SUPPORTED_H5_TYPES[t] = cls
        return cls

    return cls_wrapper


def _get_all_numpy_dtypes():
    def recurse(cls):
        subclasses = cls.__subclasses__()
        if not subclasses:
            yield cls
        for subcls in subclasses:
            yield from recurse(subcls)

    yield from recurse(np.number)


@register_h5_type(np.ndarray, *_get_all_numpy_dtypes())
class NumpyArrayH5Type:
    """Class for saving and loading numpy arrays to and from the H5 checkpoint."""

    # DEPRECATED: Do not use
    @staticmethod
    def save(ndarray, f: h5.File, key: str, **kwargs):
        """Saves a numpy array to the provided H5 file."""
        shape = tuple(ndarray.shape)
        compress_opts = kwargs if shape else {}
        dset = f.require_dataset(
            key, shape, ndarray.dtype, exact=True, **compress_opts
        )
        dset[...] = ndarray
        dset.attrs["is_bfloat16"] = is_bf16(ndarray)

    @staticmethod
    def load(f: h5.File, key: str):
        """Load a deferred numpy array."""
        from cerebras.appliance.storage.serializers import DeferredNumpyArray

        return DeferredNumpyArray(
            f.filename,
            key,
            metadata={"__TYPE__": "NumpyArraySerializer", **dict(f[key].attrs)},
        )


@register_h5_type(bool, int, float)
class ScalarH5Type:
    """
    Class for saving and loading python numeric scalars to and from the H5
    checkpoint.
    """

    @staticmethod
    def save(obj, f: h5.File, key: str, **kwargs):
        """Saves a python scalar to the provided H5 file."""
        val = np.array(obj)
        dset = f.require_dataset(key, (), val.dtype, exact=True)
        dset[...] = val

    @staticmethod
    def load(f: h5.File, key: str):
        """Loads the python scalar from the provided H5 file."""
        return f[key][...].item()


@register_h5_type(str)
class StringH5Type:
    """
    Class for saving and loading python strings to and from the H5 checkpoint.
    """

    __key__ = "string_value"

    @staticmethod
    def save(obj, f: h5.File, key: str, **kwargs):
        """Saves a python string to the provided H5 file."""
        dset = f.create_dataset(key, data=h5.Empty("f"))
        dset.attrs[StringH5Type.__key__] = obj

    @staticmethod
    def load(f: h5.File, key: str):
        """Loads the python string from the provided H5 file."""
        return f[key].attrs[StringH5Type.__key__]


@register_h5_type(type(None))
class NoneH5Type:
    """
    Class for saving and loading python Nones to and from the H5 checkpoint.
    """

    @staticmethod
    def save(obj, f: h5.File, key: str, **kwargs):
        """Saves a None placeholder to the provided H5 file."""
        f.create_dataset(key, data=h5.Empty("f"))

    @staticmethod
    def load(f: h5.File, key: str):
        """Returns a NoneType object."""
        return None


@register_h5_type(type)
class ObjectH5Type:
    """
    Fallback class for saving and loading arbitrary python objects to and from
    the H5 checkpoint.
    """

    __key__ = "object_value"

    @staticmethod
    def save(obj, f: h5.File, key: str, **kwargs):
        """Saves an arbitrary python object to the provided H5 file."""
        dset = f.create_dataset(key, data=h5.Empty("f"))
        dset.attrs[ObjectH5Type.__key__] = dill.dumps(obj).hex()

    @staticmethod
    def load(f: h5.File, key: str):
        """Load a deferred object."""
        from cerebras.appliance.storage.serializers import DeferredObject

        return DeferredObject(
            f.filename,
            key,
            metadata={"__TYPE__": "ObjectSerializer"},
        )


class UnknownH5TypeError(Exception):
    """Raised when an unknown H5 type is encountered."""
