# Copyright 2016-2023 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

"""Logging facilities for the package.

This module provides logging utilities for this package. It can be controlled
using python's default logging module.

It adds two new log levels, `TRACE` and `VERBOSE`. `VERBOSE` is more verbose
than `INFO` but less `VERBOSE` than `DEBUG`, while `TRACE` is more verbose than
`DEBUG`. Any loggers that are children of the "cerebras" logger (whether created
before this module is imported or after) will have these levels available.
As such, any child loggers can make use of these levels as follows:

    >>> import logging
    >>> logger = logging.getLogger("cerebras.foo.bar")
    >>> logger.trace("some trace message")  # doctest: +SKIP
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    AttributeError: 'Logger' object has no attribute 'trace'
    >>> import cerebras.appliance
    >>> logger = logging.getLogger("cerebras.foo.bar")
    >>> logger.setLevel(TRACE)
    >>> logger.trace("This prints with TRACE level")
    >>> logger.verbose("So does this, but with VERBOSE level")

By default, the root logger for this package is set to `WARNING`, which means
only `WARNING` and higher level messages will be enabled. In order to enable
other levels, you must set the level of the logger through one of the following:
    1. By using the root logger directly

        >>> from cerebras.appliance import logger
        >>> logger.setLevel(logging.DEBUG)

    2. By setting the level of the logger for a specific namespace

        >>> logging.getLogger(f"{ROOTNAME}").setLevel(logging.DEBUG)

    3. By setting the level of the logger for a specific sub-namespace

        >>> from cerebras.appliance import log
        >>> logging.getLogger(f"{ROOTNAME}.foo").setLevel(log.VERBOSE)


Attributes:
    ROOTNAME (str): The root logger name for this package. All loggers in this
        package should be children of this logger.
    WSC_ROOTNAME (str): The root logger name for Wafer-Scale Cluster servers.
    CRITICAL (int): The integer value of the CRITICAL log level.
    FATAL (int): The integer value of the FATAL log level.
    ERROR (int): The integer value of the ERROR log level.
    WARNING (int): The integer value of the WARNING log level.
    INFO (int): The integer value of the INFO log level.
    VERBOSE (int): The integer value of the VERBOSE log level.
    DEBUG (int): The integer value of the DEBUG log level.
    TRACE (int): The integer value of the TRACE log level.
    NOTSET (int): The integer value of the NOTSET log level.
    logger (_CerebrasLogger): The root logger for this package. This is a
        convenience variable for the logger named by `ROOTNAME`.
    wsc_logger (_CerebrasLogger): The root logger for Wafer-Scale Cluster
        servers. This is a convenience variable for the logger named by
        `WSC_ROOTNAME`.
"""

import copy
import logging
import platform
import re
from dataclasses import dataclass
from types import MethodType
from typing import (
    Callable,
    ClassVar,
    List,
    Literal,
    Optional,
    Protocol,
    Type,
    TypeVar,
    Union,
)

# Adjust stacklevel offset to return correct caller names and line numbers.
_VERSION = [int(c) for c in platform.python_version_tuple()]
ADJUST_STACKLEVEL = True
# Python 3.8 or higher
if _VERSION[0] >= 3 and _VERSION[1] >= 8:
    # Python 3.11 or higher
    if _VERSION[1] >= 11:
        STACKLEVEL_OFFSET = 2
    else:
        STACKLEVEL_OFFSET = 1
else:
    STACKLEVEL_OFFSET = 0
    ADJUST_STACKLEVEL = False

# The root logger name for this package. All loggers in this package should
# be children of this logger, so they can be controlled by setting the level
# of this logger name. You can import `logger` from this module to get the
# root logger for this package.
ROOTNAME = "cerebras"

# The root logger name for the Wafer-Scale Cluster. Loggers under this namespace
# are special in that they control the logging level of servers running on the
# Wafer-Scale Cluster.
WSC_ROOTNAME = f"{ROOTNAME}.wsc"

# Log levels defined by this package.
CRITICAL = logging.CRITICAL
FATAL = logging.FATAL
ERROR = logging.ERROR
WARNING = logging.WARNING
INFO = logging.INFO
VERBOSE = logging.INFO - 5  # i.e., levelno == 15
DEBUG = logging.DEBUG
TRACE = logging.DEBUG - 5  # i.e., levelno == 5
NOTSET = logging.NOTSET


# Set initial level to WARNING if it's not already set.
logger: "_CerebrasLogger" = logging.getLogger(ROOTNAME)
wsc_logger: "_CerebrasLogger" = logging.getLogger(WSC_ROOTNAME)
# TODO: Re-enable once all logs have been converted modelzoo is set up properly.
#       For now, we'll just let the root logger's level to decide what to log.
# if logger.level == logging.NOTSET:
#     logger.setLevel(logging.WARNING)


def get_level_name(level: Union[int, str]) -> Union[int, str]:
    """Return the textual or numeric representation of logging level 'level'.

    This is a wrapper around `logging.getLevelName` that also handles the custom
    levels defined by this module.

    Args:
        level: The textual or numeric level to get the name of.

    Returns:
        The numeric or textual representation of the level.
    """
    if isinstance(level, str):
        level = level.upper()

    if level == "VERBOSE":
        return VERBOSE
    if level == "TRACE":
        return TRACE
    if level == VERBOSE:
        return "VERBOSE"
    if level == TRACE:
        return "TRACE"
    return logging.getLevelName(level)


class ClassLogger(Protocol):
    """Protocol for an object that provides a logger and logging methods.

    The `named_class_logger` decorator injects methods/attributes into a class
    that conforms to this protocol.

    Attributes:
        logger (_CerebrasLogger): The logger for this class. This is injected by
            the `named_class_logger` decorator.
    """

    # The logger for this class.
    logger: ClassVar["_CerebrasLogger"]

    @classmethod
    def _should_log_info(cls) -> bool:
        """Returns True if the logger is enabled for INFO level."""

    @classmethod
    def _should_log_verbose(cls) -> bool:
        """Returns True if the logger is enabled for VERBOSE level."""

    @classmethod
    def _should_log_debug(cls) -> bool:
        """Returns True if the logger is enabled for DEBUG level."""

    @classmethod
    def _should_log_trace(cls) -> bool:
        """Returns True if the logger is enabled for TRACE level."""


_T = TypeVar("_T")


def named_class_logger(
    cls_or_namespace: Union[Type[_T], str, None] = None
) -> Union[Type[_T], Callable[[Type[_T]], Type[_T]]]:
    """Decorator that identifies the decorated class' logging namespace.

    Args:
        cls_or_namespace: The class to decorate or the namespace to use for the
            class' logger.
    """
    if isinstance(cls_or_namespace, type):
        return _named_class_logger(cls_or_namespace)
    elif cls_or_namespace is None or isinstance(cls_or_namespace, str):

        def _decorator(cls: Type[_T]) -> Type[_T]:
            return _named_class_logger(cls, cls_or_namespace)

        return _decorator
    else:
        raise TypeError(
            f"`cls_or_namespace` must be a type, str, or None. "
            f"Got `{type(cls_or_namespace)}`."
        )


def _named_class_logger(cls: Type[_T], name: Optional[str] = None) -> Type[_T]:
    """Decorator that identifies the decorated class' logging namespace.

    Args:
        cls: The class to inject with a logger.
        name: The name of the logger to inject. If not provided, the name will
            be inferred from the class.

    Returns:
        The passed in class injected with the logger.
    """
    # Note: Keep these in sync with the ClassLogger protocol.
    cls.logger = logging.getLogger(_qual_logger_name_for_cls(cls, name))
    cls._should_log_info = classmethod(  # pylint: disable=protected-access
        lambda c: c.logger.isEnabledFor(logging.INFO)
    )
    cls._should_log_verbose = classmethod(  # pylint: disable=protected-access
        lambda c: c.logger.isEnabledFor(VERBOSE)
    )
    cls._should_log_debug = classmethod(  # pylint: disable=protected-access
        lambda c: c.logger.isEnabledFor(logging.DEBUG)
    )
    cls._should_log_trace = classmethod(  # pylint: disable=protected-access
        lambda c: c.logger.isEnabledFor(TRACE)
    )
    return cls


class _CerebrasLogger(logging.Logger):
    """Interface for the loggers used by this package."""

    def verbose(self, msg, *args, **kwargs):
        """Log msg with severity 'VERBOSE'."""

    def trace(self, msg, *args, **kwargs):
        """Log msg with severity 'TRACE'."""


def _qual_logger_name_for_cls(
    cls: Type[ClassLogger], name: Optional[str] = None
) -> str:
    """Returns the qualified logger name for a class.

    Args:
        cls: The class to get the logger name for.
        name: The name to set the logger to. If not provided, the class'
            module/name will be used.

    Returns:
        The qualified logger name for the class.
    """
    if name:
        if name.startswith(f"{ROOTNAME}."):
            return name
        return f"{ROOTNAME}.{name}"
    return f"{ROOTNAME}.{cls.__module__}.{cls.__name__}"


@dataclass
class _CustomLevel:
    """Dataclass for custom log levels.

    Args:
        level_name: The name of the level, e.g., "TRACE".
        level_num: The integer value of the level, e.g., 5.
        method_name: The name of the method to add to the logger, e.g., "trace".
            If not provided, the lowered `level_name` will be used.
    """

    level_name: str
    level_num: int
    method_name: str = ""

    def __post_init__(self):
        """Validates the instance."""

        if self.level_num <= logging.NOTSET:
            raise ValueError(
                f"`level_num` must be greater than {logging.NOTSET}, but got "
                f"{self.level_num}"
            )

        if not self.method_name:
            if not self.level_name.isupper():
                raise ValueError(
                    f"`level_name` must be uppercase when `method_name` is not "
                    f"provided, but got `{self.level_name}`"
                )
            self.method_name = self.level_name.lower()

        if self.level_name == self.method_name:
            raise ValueError(
                f"`level_name` and `method_name` must be different, but got "
                f"`{self.level_name}` for both."
            )

        if hasattr(logging, self.level_name):
            raise ValueError(
                f"`{self.level_name}` is already defined in logging module."
            )
        if hasattr(logging, self.method_name):
            raise ValueError(
                f"`{self.method_name}` is already defined in logging module."
            )
        if hasattr(logging.getLoggerClass(), self.method_name):
            raise ValueError(
                f"`{self.method_name}` is already defined in the logger class"
            )


def _register_custom_levels(rootname: str, levels: List[_CustomLevel]) -> None:
    """Register custom logging levels to be used by children of `rootname`.

    This function registers new logging levels to all loggers which are
    descendents of the `rootname`. This includes any existing loggers prior to
    this function call as well as any new logger instances that are created
    afterwards.

    Args:
        rootname: Name of the root logger to which the new levels will be added.
        levels: List of custom levels to be added.
    """
    if not levels:
        return

    levels = copy.deepcopy(levels)

    def _extend_logger_with_level(
        alogger: logging.Logger, level: _CustomLevel
    ) -> None:
        """Extend a logger with custom levels."""

        def log_for_level(alogger: logging.Logger, message, *args, **kwargs):
            if ADJUST_STACKLEVEL:
                kwargs["stacklevel"] = (
                    kwargs.get("stacklevel", 1) + STACKLEVEL_OFFSET
                )
            alogger.log(level.level_num, message, *args, **kwargs)

        setattr(alogger, level.method_name, MethodType(log_for_level, alogger))

    def _is_package_logger(name: str) -> bool:
        return name == rootname or name.startswith(f"{rootname}.")

    def _extend_logger_with_levels(name, alogger):
        """Extend the logger if it's a descendent of `rootname`."""
        if _is_package_logger(name) and isinstance(alogger, logging.Logger):
            alogger.makeRecord = MethodType(makeRecord, alogger)
            for level in levels:
                _extend_logger_with_level(alogger, level)

    def makeRecord(self, *args, **kwargs):
        """Give a proper level name to the custom level numbers."""
        record = logging.Logger.makeRecord(self, *args, **kwargs)
        for level in levels:
            if record.levelno == level.level_num:
                record.levelname = level.level_name
                break
        return record

    # pylint: disable=not-callable
    old_getLogger = logging.Logger.manager.getLogger

    def getLogger(self, name):
        """Extend any new loggers with custom levels."""
        alogger = old_getLogger(name)
        _extend_logger_with_levels(name, alogger)
        return alogger

    # Accessing `loggerDict` requires the lock to be acquired.
    logging._acquireLock()  # pylint: disable=protected-access
    try:
        # Extend all existing loggers with custom levels.
        for name, alogger in logging.Logger.manager.loggerDict.items():
            _extend_logger_with_levels(name, alogger)

        # Set up the manager to extend any new loggers with custom levels.
        logging.Logger.manager.getLogger = MethodType(
            getLogger, logging.Logger.manager
        )
    finally:
        logging._releaseLock()  # pylint: disable=protected-access


# Register VERBOSE and TRACE logging levels to all existing and upcoming loggers
# in this package.
_register_custom_levels(
    ROOTNAME, [_CustomLevel("VERBOSE", VERBOSE), _CustomLevel("TRACE", TRACE)]
)


@dataclass
class WscLogSetting:
    """An individual logging setting for a WSC server.

    Args:
        level: The logging level to apply.
        tag: The tag of the logger to apply the level to. If not provided,
            the level will be applied as the default level.
        mode: The execution mode to apply the level to. If not provided, the
            level will be applied to all execution modes.
        role: The role to apply the level to. If not provided, the level will
            be applied to all roles.
    """

    level: int
    tag: Optional[str] = None
    mode: Optional[Literal["compile", "execute"]] = None
    role: Optional[Literal["crd", "wrk", "wgt", "act", "cmd", "br", "chf"]] = (
        None
    )

    def __post_init__(self):
        if not isinstance(self.level, int) or self.level <= logging.NOTSET:
            raise ValueError(
                f"`level` must be an integer greater than {logging.NOTSET}, "
                f"but got {self.level}."
            )
        if self.mode is not None and self.mode not in self.allowed_modes():
            raise ValueError(
                f"`mode` must be one of {self.allowed_modes()}, but got "
                f"{self.mode}."
            )
        if self.role is not None and self.role not in self.allowed_roles():
            raise ValueError(
                f"`role` must be one of {self.allowed_roles()}, but got "
                f"{self.role}."
            )

    @staticmethod
    def allowed_modes() -> List[str]:
        """Returns the allowed execution modes."""
        return ["compile", "execute"]

    @staticmethod
    def allowed_roles() -> List[str]:
        """Returns the allowed roles."""
        return ["crd", "wrk", "wgt", "act", "cmd", "br", "chf"]

    def __lt__(self, other: "WscLogSetting"):
        """Custom comparator for sorting WscLogSettings.

        This comparator sorts WscLogSettings by order of specificity. A setting
        that is applied to a single mode is considered more specific than a
        setting that is applied to all modes.

        Args:
            other: The other WscLogSetting to compare to.
        """
        if not isinstance(other, WscLogSetting):
            return NotImplemented

        self_keys = []
        other_keys = []

        attrs = ["mode", "role", "tag"]
        for attr in attrs:
            self_val = getattr(self, attr)
            other_val = getattr(other, attr)
            if self_val is None and other_val is not None:
                return True
            if self_val is not None and other_val is None:
                return False
            self_keys.append(self_val)
            other_keys.append(other_val)

        return self_keys < other_keys


def collect_wsc_log_settings() -> List[WscLogSetting]:
    """Collect all WSC server log settings.

    This method looks at all loggers that start with `wsc_logger.name` and
    collects the log level applied to them. It filters out loggers that don't
    conform to the expected format.

    Returns:
        A sorted list of WscLogSetting objects, sorted by generic to specific.
    """

    def _invalid_name_warning(name: str):
        logger.warning(
            f"Invalid Wafer-Scale Cluster server log specification: `{name}`. "
            f"This log setting has no effect and will be ignored. "
            f"Expected naming format for setting WSC server log settings is "
            f"{wsc_logger.name}.<mode>.<role>.<tag>, where mode is one of "
            f"{modes}, role is one of {roles}, and tag is the name of the "
            f"logger on the WSC server. <mode>, <role>, and <tag> are optional "
            f"(absence indicates all) but the relative ordering must be "
            f"preserved."
        )

    roles = "|".join(WscLogSetting.allowed_roles())
    modes = "|".join(WscLogSetting.allowed_modes())
    pattern = re.compile(
        fr"{wsc_logger.name}"  # logger prefix
        fr"(?:\.<(?P<mode>{modes})>)?"  # optional mode
        fr"(?:\.<(?P<role>{roles})>)?"  # optional role
        fr"(?:\.(?P<tag>[^<>]+))?"  # optional tag
    )

    logging._acquireLock()  # pylint: disable=protected-access
    try:
        # Extend all existing loggers with custom levels.
        log_settings = []
        for name, alogger in logging.Logger.manager.loggerDict.items():
            if (
                not isinstance(alogger, logging.Logger)
                or alogger.level == NOTSET
                or (
                    name != wsc_logger.name
                    and not name.startswith(f"{wsc_logger.name}.")
                )
            ):
                continue

            mobj = pattern.fullmatch(name)
            if mobj:
                log_settings.append(
                    WscLogSetting(
                        level=alogger.level,
                        tag=mobj.group("tag"),
                        mode=mobj.group("mode"),
                        role=mobj.group("role"),
                    )
                )
            else:
                _invalid_name_warning(name)
    finally:
        logging._releaseLock()  # pylint: disable=protected-access

    # Sort by generic to specific.
    log_settings.sort()

    return log_settings
