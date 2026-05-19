# Copyright 2016-2025 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import enum
import logging
import os
import time
import urllib
import uuid
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path
from threading import Lock
from typing import Optional

import h5py as h5
import hdf5plugin
import numpy

from cerebras.appliance.storage import (
    StorageDeleter,
    StorageReader,
    StorageWriter,
)
from cerebras.appliance.utils._contexts import ValueContext
from cerebras.appliance.utils.file import CachingFileOpener


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


DEFAULT_COMPRESSION: Compression = Compression.LZ4
hdf5_locking = ValueContext("best-effort")


_CACHED_FILE_OPENER = CachingFileOpener()


class H5Interface:
    def __init__(self, path_to_h5_file: str):
        self._fp = path_to_h5_file

    @classmethod
    def is_valid_path(cls, path: str) -> bool:
        has_scheme = urllib.parse.urlparse(path).scheme != ""
        return not has_scheme

    @classmethod
    def path_exists(cls, path: str) -> bool:
        return os.path.exists(path)

    @contextmanager
    def open(self, mode):
        with h5.File(self._fp, mode, locking=hdf5_locking.value) as f:
            yield f


class H5Reader(H5Interface, StorageReader):
    def __init__(
        self, path: str, compression: Compression = DEFAULT_COMPRESSION
    ):
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} does not exist")
        if not h5.is_hdf5(path):
            raise ValueError(f"{path} is not an HDF5 file")

        path = Path(path)

        StorageReader.__init__(self, path)
        H5Interface.__init__(self, path)

        # We keep the file open to avoid it being deleted from under us. This reader is
        # generally used for lazily loading values from checkpoints. By keeping this reference,
        # we're avoiding the case where during training someone deletes the original checkpoint
        # then we go and save this tensor to a new checkpoint but fail because the original was
        # deleted.
        # Note: H5 doesn't allow keeping a file open in different modes. So to keep the file open,
        # we use a regular `open()` instead of `h5.File()`.
        self._fp = _CACHED_FILE_OPENER(path, "rb")

    @property
    def fstat(self):
        return os.fstat(self._fp.fileno())

    @property
    def stats(self) -> StorageReader.Stats:
        return StorageReader.Stats(
            path=self.path,
            st_mtime=self.fstat.st_mtime,
        )

    def read(self, key):
        with self.open("r") as f:
            if key is None:  # Get global metadata
                return None, dict(f.attrs)

            dset = f[key]

            object = None
            metadata = dict(dset.attrs)
            if (void_bytes := dset.attrs.get("__BYTES__")) is not None:
                object = BytesIO(void_bytes.tobytes())

            # For legacy support only
            elif (
                dset.attrs.get("__H5_TYPE__") == "ObjectH5Type"
                and (object_value := dset.attrs.get("object_value")) is not None
            ):
                object = BytesIO(bytes.fromhex(object_value))

            elif (
                dset.attrs.get("__NUMPY__")
                # For legacy support only
                or dset.attrs.get("__H5_TYPE__")
                in ("TorchTensorH5Type", "NumpyArrayH5Type")
            ):
                object = dset[...]

            return object, metadata


class H5Writer(H5Interface, StorageWriter):
    def __init__(
        self, path: str, compression: Compression = DEFAULT_COMPRESSION
    ):
        StorageWriter.__init__(self, path)

        # atomic checkpoint save by first writing to a temp file, then renaming
        self.temp_path = Path(f"{path}.{str(uuid.uuid4())[:8]}.tmp")
        H5Interface.__init__(self, self.temp_path)

        self.compression = compression

        if compression == Compression.NONE:
            self.compress_opts = {}
        elif compression == Compression.LZ4:
            self.compress_opts = hdf5plugin.LZ4()
        else:
            self.compress_opts = {
                "compression": compression,
                "compression_opts": Compression.get_compression_options(
                    compression
                ),
            }

        self.lock = Lock()

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            # Move the temporary file to the final location
            # only if there were no exceptions
            self.temp_path.rename(self.path)

    def write(self, key, object, metadata):
        # Write with retries
        try:
            attempts = 3
            delay = 1
            while attempts:
                try:
                    return super().write(key, object, metadata)
                except Exception as e:  # pylint: disable=broad-except
                    if isinstance(e, OSError):
                        pass
                    elif isinstance(e.__cause__, OSError):
                        # PyTorchH5Saver.save_tensor() sometimes catches errors and
                        # raises a new error with the original error as the cause.
                        e = e.__cause__
                    else:
                        raise

                    attempts -= 1
                    # See note below on special handling of errno 28.
                    # If we attempt again, it will cause a segfault when file
                    # is closed, causing a hang.
                    if not attempts or e.errno == 28:
                        raise
                    logging.warning(
                        f"Will retry write in {delay} seconds. "
                        f"Remaining attempts: {attempts}",
                        exc_info=True,
                    )
                    time.sleep(delay)
                    delay *= 2
        except OSError as e:
            # HDF5 has a bug when it hits errno 28 (i.e., "No space left on device")
            # in that it runs into a segfault when closing the open file after OS
            # raises errno 28. Additionally, multiprocessing.pool fails to detect
            # child processes that just die and hangs indefinitely. These 2 issues
            # together cause a hang when we run out of disk space during checkpoint
            # saving. The open HDF5 file is closed _after_ the exception is handled
            # since we open the file for write in a context manager in
            # `PyTorchH5Saver.save_tensor()` and the file is closed in the __exit__()
            # handler. If we can somehow communicate this failure to the master process
            # _before_ the file is closed, the master process knows about the failure
            # and can exit the pool immediately and avoid the hang.
            # In the default case, we catch exceptions, serialize them, and return normally.
            # But by letting the exception pass through in the case of errno 28, the
            # multiprocessing library turns it into a `multiprocessing.pool.RemoteTraceback`
            # and sends it to the master process before the file is closed, working around
            # the hang.
            if e.errno == 28:
                logging.error(f"Ran into error writing tensor \"{key}\": {e}")
            raise

    def write_numpy(self, key, array, metadata):
        # TODO: Perhaps we can write arrays to a separate file to avoid
        #       bottlenecking
        with self.lock, self.open("a") as f:
            shape = tuple(array.shape)
            compress_opts = self.compress_opts if shape else {}

            dset = f.require_dataset(
                key,
                shape,
                array.dtype,
                exact=True,
                **compress_opts,
            )
            dset[...] = array
            dset.attrs.update(metadata)
            dset.attrs["__NUMPY__"] = True

    def write_bytes(self, key, bytes, metadata):
        with self.lock, self.open("a") as f:
            dset = f.create_dataset(key, data=h5.Empty("f"), track_order=True)
            dset.attrs.update(metadata)
            dset.attrs["__BYTES__"] = numpy.void(bytes.getbuffer())

    def write_metadata(self, key, metadata):
        with self.lock, self.open("a") as f:
            if key is None:  # Write global metadata
                f.attrs.update(metadata)
            else:
                dset = f.require_dataset(key, data=h5.Empty("f"))
                dset.attrs.update(metadata)

    @property
    def reader(self) -> type:
        return H5Reader


class H5Deleter(H5Interface, StorageDeleter):
    def __init__(self, path: str):
        StorageDeleter.__init__(self, path)
        H5Interface.__init__(self, path)

    def delete(self, key: Optional[str] = None):
        if key is not None:
            with self.lock, self.open("a") as f:
                if key in f:
                    del f[key]
                else:
                    return False
        else:
            Path(self.path).unlink(missing_ok=True)

        return True
