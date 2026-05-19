# Copyright 2016-2023 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

"""
This file contains various error types raised and error messaging during an appliance run.
"""

import copyreg
import sys
from contextlib import nullcontext

import grpc


def cerebras_support_msg(msg: str) -> str:
    """Helper function for unrecoverable messages requiring Cerebras Support"""
    return (
        f"{msg}"
        f"\nWe have encountered an internal error from Cerebras Software Stack. "
        f"For further help, "
        f"please reach out to Cerebras support team at support@cerebras.net"
    )


class ApplianceClientException(Exception):
    """Base error class for all appliance errors."""


class ApplianceUnknownError(ApplianceClientException):
    """Unknown error during execution."""


class ApplianceDeadlockError(ApplianceClientException):
    """Deadlock error."""


# Alias to maintain backwards compatibility
ApplianceStallError = ApplianceDeadlockError


class ApplianceNanError(ApplianceClientException):
    """Nan/Inf detected in outputs."""


class ApplianceTensorDropped(ApplianceClientException):
    """Tensor was dropped at appliance."""


class ApplianceDropModeComplete(ApplianceClientException):
    """Drop mode complete."""


class ApplianceRequestCancelled(ApplianceClientException):
    """Request cancelled before completion."""


class ApplianceCompilationError(ApplianceClientException):
    """Compilation error"""


class ApplianceRuntimeServerError(ApplianceClientException):
    """Runtime server (ACT, CMD, WGT) crashed/killed"""


class ApplianceServiceInterrupted(ApplianceClientException):
    """Service interrupted/aborted"""


class ApplianceRuntimeJobTimeoutError(ApplianceClientException):
    """Job time out."""


class ApplianceVersionError(ApplianceClientException):
    """Appliance software versions don't align."""


class ApplianceCsctlError(ApplianceClientException):
    """Appliance error when invoking csctl"""


class ApplianceUserVenvClusterVolumeError(ApplianceClientException):
    """Appliance error retrieving the user venv cluster volume"""


class ApplianceResourceExhausted(ApplianceClientException):
    """Resources such as memory and/or thread quotas are exhausted."""


class ClusterJobCancelledByCsctl(ApplianceClientException):
    """Appliance job cancelled by external csctl."""


class ClusterJobScheduleError(ApplianceClientException):
    """Appliance cluster lacks capacity to schedule job."""

    def __init__(self, job_id: str, message: str):
        super().__init__(message)
        self.job_id = job_id
        self.message = message


class ClusterJobInitError(ApplianceClientException):
    """Appliance job failed during initialization."""

    def __init__(self, job_id: str, message: str):
        super().__init__(message)
        self.job_id = job_id
        self.message = message


class PicklableRpcError(grpc.RpcError, grpc.Call, grpc.Future):
    """A picklable exception substitute for a generic gRPC error.

    gRPC errors are not guaranteed to be picklable. There are portions of our
    code that run in subprocesses and communicate through gRPC. Any exceptions
    raised in those subprocesses go through pickling to be sent to the main
    process. To support this flow with gRPC exceptions, we register custom
    picklers for them. The unpickled exception is this object.

    This class provides an interface that matches that of gRPC errors. However,
    it is not a complete substitue for gRPC exceptions and should not be used
    for anything other than basic logging/instance checking purposes.
    """

    class _RPCState:
        def __init__(
            self,
            initial_metadata,
            trailing_metadata,
            code,
            details,
            debug_error_string="",
        ):
            self.initial_metadata = initial_metadata
            self.trailing_metadata = trailing_metadata
            self.code = code
            self.details = details
            self.debug_error_string = debug_error_string
            self.condition = nullcontext()

    # pylint: disable=super-init-not-called
    def __init__(self, exc_class_name, *args, **kwargs):
        self._exc_class_name = exc_class_name
        self._state = self._RPCState(*args, **kwargs)

    @classmethod
    def from_grpc_error(cls, obj: grpc.RpcError):
        """Creates a `PicklableRpcError` from a gRPC error."""
        return cls(
            obj.__class__.__name__,
            obj.initial_metadata(),
            obj.trailing_metadata(),
            obj.code(),
            obj.details(),
            debug_error_string=getattr(
                obj, "debug_error_string", lambda: None
            )(),
        )

    def _repr(self):
        formatted_exception = (
            f"gRPC Error:\n"
            f"  Status Code: {self.code()}\n"
            f"  Details: {self.details()}\n"
        )

        if trailing_metadata := self.trailing_metadata():
            formatted_exception += "  Metadata:\n"
            for key, value in trailing_metadata:
                formatted_exception += f"    {key}: {value}\n"

        return formatted_exception

    def __repr__(self):
        return self._repr()

    def __str__(self):
        return self._repr()

    def initial_metadata(self):
        """Accesses the initial metadata sent by the server."""
        return self._state.initial_metadata

    def trailing_metadata(self):
        """Accesses the trailing metadata sent by the server."""
        return self._state.trailing_metadata

    def code(self):
        """Accesses the status code sent by the server."""
        return self._state.code

    def details(self):
        """Accesses the details sent by the server."""
        return self._state.details

    def cancel(self):
        """See grpc.Future.cancel."""
        return False

    def cancelled(self):
        """See grpc.Future.cancelled."""
        return False

    def running(self):
        """See grpc.Future.running."""
        return False

    def done(self):
        """See grpc.Future.done."""
        return True

    def result(self, timeout=None):
        """See grpc.Future.result."""
        raise self

    def exception(self, timeout=None):
        """See grpc.Future.exception."""
        return self

    def traceback(self, timeout=None):
        """See grpc.Future.traceback."""
        try:
            raise self
        except grpc.RpcError:
            return sys.exc_info()[2]

    def is_active(self):
        """Describes whether the RPC is active or has terminated."""
        return False

    def time_remaining(self):
        """Describes the length of allowed time remaining for the RPC."""
        return 0

    def add_callback(self, callback):
        """Registers a callback to be called on RPC termination."""
        return False

    def add_done_callback(self, fn):
        """See grpc.Future.add_done_callback."""
        fn(self)

    def __iter__(self):
        return self

    def __next__(self):
        raise self

    def next(self):
        """Returns next item in iterator."""
        return self.__next__()


def register_grpc_error_pickler():
    """Register custom pickle/unpickle handlers for gRPC exceptions."""
    from cerebras.appliance.utils.classes import retrieve_all_subclasses

    for rpc_exc_cls in retrieve_all_subclasses(grpc.RpcError):
        if not issubclass(rpc_exc_cls, PicklableRpcError):
            copyreg.pickle(rpc_exc_cls, _grpc_error_pickler)


def _grpc_error_pickler(obj: grpc.RpcError):
    """Custom pickler for gRPC exceptions."""
    return (
        _grpc_error_unpickler,
        (
            (
                obj.__class__.__name__,
                obj.initial_metadata(),
                obj.trailing_metadata(),
                obj.code(),
                obj.details(),
            ),
            {
                "debug_error_string": getattr(
                    obj, "debug_error_string", lambda: None
                )()
            },
            obj.__cause__,
            obj.__traceback__,
        ),
    )


def _grpc_error_unpickler(args, kwargs, cause, traceback) -> PicklableRpcError:
    """Custom unpickler for gRPC exceptions."""
    inst = PicklableRpcError(*args, **kwargs)
    inst.__cause__ = cause
    inst.__traceback__ = traceback
    return inst
