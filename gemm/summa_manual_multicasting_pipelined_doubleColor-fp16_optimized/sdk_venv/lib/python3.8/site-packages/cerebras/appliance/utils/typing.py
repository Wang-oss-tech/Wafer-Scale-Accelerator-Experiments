#
# CC BY-SA 4.0 from Daniil Fajnberg
# From https://stackoverflow.com/a/73966151/381313
#
# Copyright 2016-2023 Cerebras Systems
# SPDX-License-Identifier: CC-BY-SA-4.0
#
"""Utility functions for working with type hints."""

from collections.abc import Callable
from inspect import Parameter, Signature
from typing import Any, Literal, Union, get_args, get_origin


def param_matches_type_hint(
    param: Parameter,
    type_hint: type,
    strict: bool = False,
) -> bool:
    """
    Returns `True` if the parameter annotation matches the type hint.

    For this to be the case:
    In `strict` mode, both must be exactly equal.
    If both are specified generic types, they must be exactly equal.
    If the parameter annotation is a specified generic type and
    the type hint is an unspecified generic type,
    the parameter type's origin must be that generic type.
    """
    param_origin = get_origin(param.annotation)
    type_hint_origin = get_origin(type_hint)
    if (
        strict
        or (param_origin is None and type_hint_origin is None)
        or (param_origin is not None and type_hint_origin is not None)
    ):
        return param.annotation == type_hint
    if param_origin is None and type_hint_origin is not None:
        return False
    return param_origin == type_hint


def signature_matches_type_hint(
    sig: Signature,
    type_hint: type,
    strict: bool = False,
) -> bool:
    """
    Returns `True` if the function signature and `Callable` type hint match.

    For details about parameter comparison, see `param_matches_type_hint`.
    """
    if get_origin(type_hint) != Callable:
        raise TypeError("type_hint must be a `Callable` type")
    type_params, return_type = get_args(type_hint)
    if sig.return_annotation != return_type:
        return False
    if len(sig.parameters) != len(type_params):
        return False
    return all(
        param_matches_type_hint(sig_param, type_param, strict=strict)
        for sig_param, type_param in zip(sig.parameters.values(), type_params)
    )


def check_type(obj: Any, type_hint: type) -> bool:
    """
    Check if the object's type matches the type hint.

    Args:
        obj: The object to check.
        type_hint: The type hint to compare against.
    """
    origin = get_origin(type_hint)
    args = get_args(type_hint)

    if origin is None:
        if type_hint is Any:
            return True

        return isinstance(obj, type_hint)

    if (
        (origin is list)
        or (origin is set)
        or (origin is tuple)
        or (origin is dict)
    ):
        if not isinstance(obj, origin):
            return False

        if not args:
            # No type hint for elements
            return True

        if origin is dict:
            return all(
                check_type(k, args[0]) and check_type(v, args[1])
                for k, v in obj.items()
            )

        return all(check_type(v, args[0]) for v in obj)

    if origin is Union:
        return any(check_type(obj, arg) for arg in args)

    if origin is Literal:
        return obj in args

    raise ValueError(f"Unsupported type hint: {type_hint}")


def type_hint_to_string(type_hint: type, include_prefix: bool = True):
    """
    Convert a type hint to a human-readable string.

    Args:
        type_hint: The type hint to convert.
        include_prefix: If `True`, include a prefix in the string.
    """
    origin = get_origin(type_hint)
    args = get_args(type_hint)

    if origin is None:
        if isinstance(type_hint, str):
            return f'"{type_hint}"'

        if name := getattr(type_hint, "__name__", None):
            return name
        if name := getattr(type_hint, "_name", None):
            return name

        return str(type_hint)

    if (origin is list) or (origin is set) or (origin is tuple):
        return f"{origin.__name__} of {type_hint_to_string(args[0])}"

    if origin is dict:
        prefix = ""
        if include_prefix:
            prefix = "dictionary of "

        return f"{prefix}{type_hint_to_string(args[0])} to {type_hint_to_string(args[1])} pairs"

    if (origin is Union) or (origin is Literal):
        prefix = ""
        if include_prefix:
            prefix = "one of "

        return prefix + ", ".join(
            str(type_hint_to_string(arg, False)) for arg in args
        )

    raise ValueError(f"Unsupported type hint: {type_hint}")
