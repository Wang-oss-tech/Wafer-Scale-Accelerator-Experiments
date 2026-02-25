# Copyright 2016-2023 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

"""
Utils file for miscellaneous functions and classes needed
"""
import logging
import os
import sys
from contextlib import contextmanager
from functools import lru_cache
from importlib.util import find_spec
from typing import Optional

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
            del os.environ[key]


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


def resolve_command_line_arg(
    command_line_arg: str,
    env_var_name: str,
    arg_description: str,
    executable_name: str,
    arg_name: str,
    required: bool = True,
    default: Optional[str] = None,
) -> Optional[str]:
    """
    Utility method for resolving command line arguments with environment variable
    fallback to start up appliance servers.

    Args:
        command_line_arg: The argument provided on the command line (if any)
        env_var_name: Name of the environment variable to check if command_line_arg is empty
        arg_description: Description of the argument to use in error messages
        executable_name: Name of the executable to use in error messages
        arg_name: Argument name for usage log
        required: Whether the argument is required (True) or optional (False)
        default: Default value to use if argument is optional and not provided

    Returns:
        The resolved argument value as a string or None
    """
    resolved_arg = command_line_arg if command_line_arg else None

    if resolved_arg is None:
        resolved_arg = os.environ.get(env_var_name)

        if resolved_arg is not None:
            logging.info(
                f"Using {env_var_name} environment variable: {resolved_arg}"
            )
        elif required:
            error_msg = (
                f"Error: {arg_description} not specified.\n"
                "Please either:\n"
                f"  1. Provide it with the {arg_name} argument to `{executable_name}`\n"
                f"  2. Set the {env_var_name} environment variable"
            )
            print(error_msg, file=sys.stderr)
            sys.exit(2)
        else:
            # For optional arguments, use the default if provided
            resolved_arg = default

    return resolved_arg


def is_true(value):
    """
    Check if a value is considered True.

    Args:
        value: The value to check.

    Returns:
        bool: True if the value is considered True, False otherwise.
    """
    if value is None:
        return False
    if isinstance(value, str):
        value = value.lower()
        if value in {"t", "true", "y", "yes"}:
            return True
        if value in {"f", "false", "n", "no"}:
            return False
        return bool(int(value))

    # pylint: disable=superfluous-parens,unnecessary-negation
    return not (not value)
