# Copyright 2016-2023 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause
"""
Utils file for creating and generating a Python traceback from coordinator compile errors
"""

from typing import Optional

from google.protobuf.json_format import Parse
from tblib import Traceback

from cerebras.appliance.pb.common.diagnostic_pb2 import ErrorMessage


def _get_tb_frame_from_stackframe(stackframe):
    """
    Given a protobuf structure of ErrorMessage's single stackframe, parse it and return the
    tblib traceback structure representing the single stackframe.
    """
    return {
        "tb_frame": {
            "f_code": {
                "co_filename": stackframe.filename,
                "co_name": stackframe.function,
            },
            "f_globals": {
                "__file__": stackframe.filename,
                "__name__": "__main__",
            },
            "f_lineno": stackframe.line,
        },
        "tb_lineno": stackframe.line,
        "tb_next": None,
    }


def _get_source_location_traceback_dictionary(traceback):
    """
    Given the protobuf structure of ErrorMessage as a traceback, parse through the entire stackframe
    and return the tblib traceback structure representing the entire stackframe as a Python
    Exception traceback.
    """
    if len(traceback) == 0:
        return None

    first_stackframe = _get_tb_frame_from_stackframe(traceback[0])
    prev_stackframe = first_stackframe

    # skip the first one
    for stackframe in traceback[1:]:
        tb_frame = _get_tb_frame_from_stackframe(stackframe)
        prev_stackframe["tb_next"] = tb_frame
        prev_stackframe = tb_frame

    return first_stackframe


def get_lowering_error_from_json(traceback_json: str) -> RuntimeError:
    """
    Given a JSON string representing the traceback, parse through it and return the RuntimeError
    representing the exception with the encoded traceback.
    """
    diag = Parse(traceback_json, ErrorMessage())

    runtime_error = RuntimeError(f"Lowering failed due to `{diag.message}`")

    traceback = _get_source_location_traceback_dictionary(diag.traceback)

    return runtime_error.with_traceback(
        Traceback.from_dict(traceback).as_traceback()
    )


def get_lowering_exception(exc: tuple) -> Optional[RuntimeError]:
    """
    Given the metadata which is a tuple of key-value pairs representing the headers attached to gRPC
    calls, return the exception created from the call if a traceback JSON object is provided.
    """

    for key, value in exc:
        if key != "cerebras-lowering-error-bin":
            continue

        # if it's the key we're looking for, create the traceback and return it
        try:
            return get_lowering_error_from_json(value)
        except:  # pylint: disable=bare-except
            return None
    return None
