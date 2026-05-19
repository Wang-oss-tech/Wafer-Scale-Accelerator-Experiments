# Copyright 2016-2023 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

"""
utils file for miscellaneous functions and classes needed
"""
import os
from contextlib import contextmanager
from functools import lru_cache
from importlib.util import find_spec

from cerebras.appliance._version import __githash__, __version__
from cerebras.appliance.errors import ApplianceVersionError


@contextmanager
def limit_mp_threads():
    """Turn off threadings parameters for multiprocessing situations"""
    thread_reductions = {
        'OPENBLAS_NUM_THREADS': '1',
        'OMP_NUM_THREADS': '1',
        'XLA_THREAD_POOL_SIZE': '1',
        'XLA_IO_THREAD_POOL_SIZE': '1',
    }
    original_env_values = {}
    additional_env_keys = []
    for key in thread_reductions:
        value = os.environ.get(key, None)
        if value is not None:
            original_env_values[key] = value
        else:
            additional_env_keys.append(key)
    try:
        os.environ.update(thread_reductions)
        yield
    finally:
        os.environ.update(original_env_values)
        for key in additional_env_keys:
            os.environ.unsetenv(key)


def version_check(external_component: str, ext_version: str, ext_githash: str):
    """Validate server version info"""
    if __githash__ == ext_githash:
        # No matter the version strings, its the same build so its compatible.
        return
    # Build mismatch of some kind.
    server_public = ext_version.split("-")[0]
    client_public = __version__.split("+")[0]
    if (
        client_public == server_public
        or server_public == "0.0.0"
        or client_public == "0.0.0"
    ):
        # Internal build mismatch
        error_msg = (
            f"Client software is version {__version__} on {__githash__} but "
            f"{external_component} version is {ext_version} on {ext_githash}."
        )
    else:
        # Release version mismatch
        error_msg = (
            f"{external_component} has version: {server_public} but client "
            f"has {client_public}.\nIn order to use this cluster, you must "
            f"install {server_public} of the client.\n"
        )
    raise ApplianceVersionError(error_msg)


@lru_cache(maxsize=1)
def is_cerebras_available():
    """Simple check for availability of internal cerebras package"""
    return find_spec("cerebras") is not None
