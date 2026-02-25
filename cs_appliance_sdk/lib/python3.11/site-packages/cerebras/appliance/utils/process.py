# Copyright 2016-2023 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

"""Utilities for process management."""

from typing import Callable, Union

import psutil


def map_processes(
    map_fn: Callable[[psutil.Process], None],
    process: Union[int, psutil.Process, None] = None,
    include_children: bool = False,
):
    """Applies the map function to the given process and all its children.

    Args:
        process: The process or process ID to visit. If None, the current
            process is used.
        include_children: Whether to include all children.
        map_fn: The function to apply to each process.
    """
    if not callable(map_fn):
        raise TypeError(
            f"Invalid map function type: {type(map_fn)}. Expected a callable."
        )

    if process is None:
        process = psutil.Process()
    elif isinstance(process, int):
        process = psutil.Process(process)

    if not isinstance(process, psutil.Process):
        raise TypeError(
            f"Invalid process type: {type(process)}. Expected a process ID or "
            f"a `psutil.Process` instance."
        )

    if map_fn is not None:
        map_fn(process)

    if include_children:
        try:
            for child in process.children(recursive=True):
                try:
                    map_fn(child)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
