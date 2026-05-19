# Copyright 2016-2023 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

"""Python class utilities"""
from typing import Callable, Generator, Optional, Type


def retrieve_all_subclasses(
    cls: Type,
    condition: Optional[Callable[[Type], bool]] = None,
) -> Generator[Type, None, None]:
    """
    Retrieves all subclasses of a given class

    Args:
        cls: The class to retrieve subclasses of
        condition: A callable that takes a class and returns whether or not to
            include it in the results. If None, all subclasses are returned.
    """
    for subcls in cls.__subclasses__():
        if condition is None or condition(subcls):
            yield subcls
        yield from retrieve_all_subclasses(subcls, condition)
