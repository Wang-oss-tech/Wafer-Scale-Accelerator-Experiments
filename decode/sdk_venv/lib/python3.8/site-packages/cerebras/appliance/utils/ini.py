# Copyright 2016-2024 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

"""
INI Class that handles setting INIs in DebugArgs
"""

from types import SimpleNamespace
from typing import Optional

from typing_extensions import Self

from cerebras.appliance.pb.workflow.appliance.common.common_config_pb2 import (
    DebugArgs,
)


class INI(SimpleNamespace):
    """A simple namespace class to handle setting INI values in DebugArgs."""

    def __init__(self, /, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __setattr__(self, name, value):
        if not isinstance(value, (bool, int, float, str)):
            raise TypeError(
                f"INI expected bool, int, float, or str for `{name}`. Got {value}"
            )

        if self._debug_args:
            set_ini(self._debug_args, **{name: value})

        return super().__setattr__(name, value)

    def __delattr__(self, name):
        if self._debug_args:
            clear_ini(self._debug_args, name)

        return super().__delattr__(name)

    @property
    def _debug_args(self) -> Optional[DebugArgs]:
        """Returns the DebugArgs instance that is tied to this INI"""
        return None

    def update(self):
        """Update the INI after setting a new value."""

    def __iter__(self):
        yield from self.__dict__.items()

    @staticmethod
    def get_supported_type_maps(debug_args: DebugArgs):
        """Get the supported type maps for INI values."""
        return {
            bool: debug_args.ini.bools,
            int: debug_args.ini.ints,
            float: debug_args.ini.floats,
            str: debug_args.ini.strings,
        }

    def propagate(self, debug_args: DebugArgs, override: bool = True):
        """Propagate the INI values to the DebugArgs."""
        ini_maps = self.get_supported_type_maps(debug_args)
        for k, v in self.__dict__.items():
            ini_map = ini_maps[type(v)]
            if k not in ini_map or override:
                ini_map[k] = v

    def clear(self, debug_args: DebugArgs):
        """Clear the INI values from the DebugArgs."""
        ini_maps = self.get_supported_type_maps(debug_args)
        for k in self.__dict__:
            for ini_map in ini_maps.values():
                ini_map.pop(k, None)

    def merge_from(self, debug_args: DebugArgs):
        """Clear the INI values from the DebugArgs."""
        ini_maps = self.get_supported_type_maps(debug_args)
        for ini_map in ini_maps.values():
            self.__dict__.update(ini_map)

    def copy_from(self, debug_args: DebugArgs):
        """Clear the INI values from the DebugArgs."""
        self.__dict__.clear()
        self.merge_from(debug_args)

    def __or__(self, ini: Self):
        if isinstance(ini, dict):
            ini = self.__class__(**ini)
        elif not isinstance(ini, INI):
            raise TypeError(f"Expected an INI, but got: {type(ini)}")

        return self.__class__(**dict(self), **dict(ini))

    def __ior__(self, ini: Self):
        if isinstance(ini, dict):
            ini = INI(**ini)
        elif not isinstance(ini, INI):
            raise TypeError(f"Expected an INI, but got: {type(ini)}")

        for k, v in ini:
            setattr(self, k, v)

        return self

    def __isub__(self, ini: Self):
        if isinstance(ini, dict):
            ini = INI(**ini)
        elif not isinstance(ini, INI):
            raise TypeError(f"Expected an INI, but got: {type(ini)}")

        for k, _ in ini:
            if hasattr(self, k):
                delattr(self, k)

        return self


def set_ini(debug_args: DebugArgs, **kwargs):
    """Set an Debug INI in the DebugArgs."""
    ini = INI(**kwargs)
    ini.propagate(debug_args, override=True)


def set_default_ini(debug_args: DebugArgs, **kwargs):
    """Set default INI in the DebugArgs, if INI is not set."""
    ini = INI(**kwargs)
    ini.propagate(debug_args, override=False)


def clear_ini(debug_args: DebugArgs, *args, **kwargs):
    """Unsets INIs in DebugArgs."""
    ini = INI(**{a: False for a in args}, **kwargs)
    ini.clear(debug_args)


get_supported_type_maps = INI.get_supported_type_maps
