# Copyright 2016-2023 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

"""Context manager utilities."""

from contextlib import contextmanager
from typing import Any


class ValueContext:
    """A context manager for temporarily changing an arbitrary value."""

    def __init__(self, default: Any):
        self.value = default

    def __repr__(self):
        return f"{self.__class__.__name__}(value={self.value}) @ {id(self):#x}"

    @contextmanager
    def __call__(self, value: Any):
        """Temporarily set the value of the context manager"""
        old_value = self.value
        self.value = value
        try:
            yield
        finally:
            self.value = old_value


class BooleanContext(ValueContext):
    """A context manager for temporarily changing a boolean value."""

    def __bool__(self):
        return self.value
