# Copyright 2016-2023 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

"""Utility functions for converting between units."""

from numbers import Number
from typing import Literal, Optional, Union

_BytePowersOf10 = Literal["B", "KB", "MB", "GB", "TB", "PB"]
_BytePowersOf2 = Literal["B", "KiB", "MiB", "GiB", "TiB", "PiB"]
ByteUnit = Union[_BytePowersOf10, _BytePowersOf2]

TimeUnit = Literal["ns", "us", "ms", "s", "m", "h", "d", "w", "y"]

# Mapping between byte units (powers of 2) and number of bytes in that unit.
_byte_power_2_units = {
    u: (1 << (i * 10))
    for i, u in enumerate(("B", "KiB", "MiB", "GiB", "TiB", "PiB"))
}

# Mapping between byte units (powers of 10) and number of bytes in that unit.
_byte_power_10_units = {
    u: (10 ** (i * 3))
    for i, u in enumerate(("B", "KB", "MB", "GB", "TB", "PB"))
}

# Mapping between time units and the number of nanoseconds in that unit.
_time_units = {
    "ns": 1,
    "us": 1e3,
    "ms": 1e6,
    "s": 1e9,
    "m": 6e10,
    "h": 3.6e12,
    "d": 8.64e13,
    "w": 6.048e14,
    "y": 3.154e16,
}


def convert_byte_unit(
    num: int,
    tgt_unit: ByteUnit,
    /,
    src_unit: ByteUnit = "B",
    precision: int = 0,
) -> Union[int, float]:
    """Convert a value in bytes to the specified unit.

    Args:
        num: The value to convert.
        tgt_unit: The unit to convert to.
        src_unit: The unit that the value is in. Defaults to "B".
        precision: The number of decimal places to round to. Defaults to 0.
    Returns:
        The value converted to the given unit.
    """
    if precision < 0:
        raise ValueError(f"Invalid precision: {precision}")

    if src_unit in _byte_power_10_units:
        src_multiplier = _byte_power_10_units[src_unit]
    elif src_unit in _byte_power_2_units:
        src_multiplier = _byte_power_2_units[src_unit]
    else:
        raise ValueError(f"Invalid `src_unit`: {src_unit}")

    if tgt_unit in _byte_power_10_units:
        tgt_multiplier = _byte_power_10_units[tgt_unit]
    elif tgt_unit in _byte_power_2_units:
        tgt_multiplier = _byte_power_2_units[tgt_unit]
    else:
        raise ValueError(f"Invalid `tgt_unit`: {tgt_unit}")

    res = float(num * src_multiplier) / tgt_multiplier
    res = round(res, precision)
    if precision == 0:
        return int(res)
    return res


def bytes_to_human(
    num: int, /, fmt: str = "{value:.1f}{unit}", log_base: int = 2
) -> str:
    """Convert a value in bytes to a human-readable string.

    Implementation inspired by `psutil._common.bytes2human`.

    Args:
        num: The value to convert.
        fmt: The format string to use. It must contain the keys `value` and
            `unit`. Defaults to "{value:.1f}{unit}".
        log_base: The base of the logarithm to use. Must be 2 or 10. Defaults
            to 2.
    Returns:
        The value converted to a human-readable string.
    """
    if log_base == 2:
        units = ("B", "KiB", "MiB", "GiB", "TiB", "PiB")
        prefix = {u: 1 << (i * 10) for i, u in enumerate(units)}
    elif log_base == 10:
        units = ("B", "KB", "MB", "GB", "TB", "PB")
        prefix = {u: 10 ** (i * 3) for i, u in enumerate(units)}
    else:
        raise ValueError(f"Invalid `log_base`: {log_base}")

    for unit in reversed(units[1:]):
        if num >= prefix[unit]:
            value = float(num) / prefix[unit]
            break
    else:
        value = num
        unit = units[0]

    return fmt.format(unit=unit, value=value)


def convert_time_unit(
    num: Number,
    src_unit: TimeUnit,
    tgt_unit: TimeUnit,
    precision: Optional[int] = None,
) -> Union[int, float]:
    """Convert a time value with some unit to the target unit.

    Args:
        num: The value to convert.
        src_unit: The unit that value is in.
        tgt_unit: The unit to convert to.
        precision: The number of decimal places to round to. It None, no
            rounding is performed. Defaults to None.
    Returns:
        The time value converted to the given unit.
    """
    if precision is not None and precision < 0:
        raise ValueError(f"Invalid precision: {precision}")

    if src_unit not in _time_units:
        raise ValueError(f"Invalid `src_unit`: {src_unit}")
    if tgt_unit not in _time_units:
        raise ValueError(f"Invalid `tgt_unit`: {tgt_unit}")

    res = float(num) * _time_units[src_unit] / _time_units[tgt_unit]
    if precision is not None:
        res = round(res, precision)
        if precision == 0:
            return int(res)
    return res


def time_to_human(
    num: Number, src_unit: TimeUnit, fmt: str = "{value:.1f}{unit}"
) -> str:
    """Convert a time value with some unit to a human-readable string.

    Args:
        num: The value to convert.
        src_unit: The unit that value is in.
        fmt: The format string to use. It must contain the keys `value` and
            `unit`. Defaults to "{value:.1f}{unit}".
    Returns:
        The time value converted to a human-readable string.
    """
    if src_unit not in _time_units:
        raise ValueError(f"Invalid `src_unit`: {src_unit}")

    num_ns = float(num) * _time_units[src_unit]
    for unit in reversed(list(_time_units.keys())[1:]):
        if num_ns >= _time_units[unit]:
            value = num_ns / _time_units[unit]
            break
    else:
        value = num_ns
        unit = list(_time_units.keys())[0]

    return fmt.format(unit=unit, value=value)
