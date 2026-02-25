# Copyright 2016-2023 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

"""Utilities for memory management."""

import functools
import logging
import re
from collections import namedtuple
from contextlib import contextmanager
from typing import Generator, List, Union

import psutil

from cerebras.appliance import log

from .process import map_processes
from .units import ByteUnit, bytes_to_human, convert_byte_unit

# Detailed view of the memory usage of a process. See `man proc` for details.
pdetailedmem = namedtuple(
    'pdetailedmem',
    (
        "RssAnon",  # Resident anonymous memory
        "RssFile",  # Resident file-backed memory
        "RssShmem",  # Resident shared memory
        "VmPeak",  # peak virtual memory size
        "VmSize",  # total program size
        "VmLck",  # locked memory size
        "VmHWM",  # peak resident set size ("high water mark")
        "VmRSS",  # size of memory portions
        "VmData",  # size of data, stack, and text segments
        "VmStk",  # size of data, stack, and text segments
        "VmExe",  # size of text segment
        "VmLib",  # size of shared library code
        "VmPTE",  # size of page table entries
        "VmSwap",  # size of swap usage (the number of referred swapents)
    ),
)


def get_available_memory(unit: ByteUnit = "B") -> int:
    """Returns the available non-swap memory that can be used by this process.

    Args:
        unit: The unit to return the memory in.
    Returns:
        The available memory in the specified unit.
    """
    return convert_byte_unit(psutil.virtual_memory().available, unit)


def get_used_memory(unit: ByteUnit = "B") -> int:
    """Returns the used system memory.

    Args:
        unit: The unit to return the memory in.
    Returns:
        The used memory in the specified unit.
    """
    return convert_byte_unit(psutil.virtual_memory().used, unit)


def get_process_memory_full_info(
    process: Union[int, psutil.Process, None] = None,
    include_children: bool = False,
    unit: ByteUnit = "B",
) -> psutil._pslinux.pfullmem:
    """Returns the most accurate estimate of used memory of the given process.

    Note that this method is quite expensive and requires higher OS previleges.

    Args:
        process: The process or process ID to get the memory usage for. If None,
            the current process is used.
        include_children: Whether to include the memory usage of all children.
        unit: The unit to return the memory in.
    Returns:
        A tuple of (uss, pss, swap) memory usage in the specified unit.
            uss: Memory that is unique to the process.
            pss: Memory that is shared with other processes.
            swap: Memory that is swapped to disk.
    """

    def map_fn(process: psutil.Process):
        nonlocal infos
        infos.append(process.memory_full_info())

    infos = []
    map_processes(map_fn, process=process, include_children=include_children)

    result = {key: 0 for key in infos[0]._fields}
    for info in infos:
        for key in result:
            result[key] += getattr(info, key)

    for key, value in result.items():
        result[key] = convert_byte_unit(value, unit)

    return type(infos[0])(**result)


def get_detailed_memory_info(
    process: Union[int, psutil.Process, None] = None,
    include_children: bool = False,
    unit: ByteUnit = "B",
) -> pdetailedmem:
    """Returns super fine-grained memory info of the given process.

    This detailed view contains info about what type of memory is used (e.g.,
    file vs anonymous). See `man proc` for details.

    Args:
        process: The process or process ID to get the memory usage for. If None,
            the current process is used.
        include_children: Whether to include the memory usage of all children.
        unit: The unit to return the memory in.
    Returns:
        A namedtuple containing the detailed memory info.
    """

    def map_fn(process: psutil.Process):
        nonlocal infos

        stat_file = f"/proc/{process.pid}/status"
        try:
            with open(stat_file, "r") as f:
                content = f.read()
        except Exception:  # pylint: disable=broad-except
            return

        patterns = {
            key: re.compile(rf"^{key}:\s*(?P<value>\d+)\s*kB")
            for key in pdetailedmem._fields
        }
        values = {key: 0 for key in pdetailedmem._fields}

        lines = content.split("\n")
        for line in lines:
            for key in pdetailedmem._fields:
                if key not in patterns:
                    continue
                mobj = patterns[key].match(line)
                if mobj:
                    # Convert to byte here for more accurate results when
                    # summing up the values from children
                    values[key] = convert_byte_unit(
                        int(mobj.group("value")), "B", src_unit="KiB"
                    )
                    del patterns[key]  # Pop key to avoid unnecessary lookups
                    break
        infos.append(pdetailedmem(**values))

    infos = []
    map_processes(map_fn, process=process, include_children=include_children)

    result = {key: 0 for key in infos[0]._fields}
    for info in infos:
        for key in result:
            result[key] += getattr(info, key)

    for key, value in result.items():
        result[key] = convert_byte_unit(value, unit)

    return type(infos[0])(**result)


@contextmanager
def with_memory_info_logged(
    msg: str,
    /,
    info: Union[str, List[str]] = "",
    logger: logging.Logger = log.logger,
    log_level: int = log.VERBOSE,
) -> Generator[None, None, None]:
    """Context manager that logs yields the used memory.

    Args:
        msg: The message postfix to log.
        info: The information to log. See `psutil.virtual_memory`,
            `psutil.memory_full_info`, `pdetailedmem` for the list of available
            options. Note that `psutil.memory_full_info` and `pdetailedmem`
            are expensive.
        unit: The unit to return the memory in.
        logger: The logger to use for logging.
        log_level: The verbosity level to use for logging.
    Yields:
        None
    """
    if isinstance(info, str):
        info = [info]

    def _log_usage(postfix: str):
        if info and logger.isEnabledFor(log_level):

            @functools.lru_cache
            def virtual_memory():
                return psutil.virtual_memory()

            @functools.lru_cache()
            def full_memory():
                return get_process_memory_full_info()

            @functools.lru_cache()
            def detailed_memory():
                return get_detailed_memory_info()

            mem_str = []
            for field in info:
                # pylint: disable=protected-access
                if field in psutil._pslinux.pfullmem._fields:
                    attr = getattr(full_memory(), field)
                # pylint: disable=protected-access
                elif field in psutil._pslinux.svmem._fields:
                    attr = getattr(virtual_memory(), field)
                elif field in pdetailedmem._fields:
                    attr = getattr(detailed_memory(), field)
                else:
                    valid_fields = (
                        psutil._pslinux.pfullmem._fields
                        + psutil._pslinux.svmem._fields
                        + pdetailedmem._fields
                    )
                    # pylint: disable=protected-access
                    raise AttributeError(
                        f"Invalid memory info field '{field}'. "
                        f"Available fields to logs are: "
                        f"{', '.join(valid_fields)}."
                    )

                if field == "percent":
                    value = f"{attr:.2f}%"
                else:
                    value = bytes_to_human(attr)
                mem_str.append(f"{field}={value}")

            mem_str = ", ".join(mem_str)
            mem_str = f"Memory usage {postfix}: {mem_str}"
            logger.log(log_level, mem_str)

    _log_usage(f"before {msg}")
    yield
    _log_usage(f"after {msg}")
