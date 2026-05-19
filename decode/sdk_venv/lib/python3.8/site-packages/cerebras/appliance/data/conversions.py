# Copyright 2016-2023 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

"""Conversion utilities between dtypes."""

from typing import Union

import numpy as np

from cerebras.appliance.data.dtypes import bf16, is_bf16
from cerebras.appliance.pb.ws.rtfx_pb2 import RtFxProto


def rtfx_dtype_from_np_dtype(np_dtype: np.dtype) -> int:
    """Converts a numpy dtype to an RtFx dtype."""

    assert isinstance(np_dtype, np.dtype), "Numpy dtype expected"
    if is_bf16(np_dtype):
        return RtFxProto.T_BF16
    elif np_dtype == bool:
        return RtFxProto.T_I1  # BUT NOTE: it needs casting
    elif np_dtype == np.int16:
        return RtFxProto.T_I16
    elif np_dtype == np.int32:
        return RtFxProto.T_I32
    elif np_dtype == np.int64:
        return RtFxProto.T_I64
    elif np_dtype == np.uint8:
        return RtFxProto.T_U8
    elif np_dtype == np.uint16:
        return RtFxProto.T_U16
    elif np_dtype == np.uint32:
        return RtFxProto.T_U32
    elif np_dtype == np.uint64:
        return RtFxProto.T_U64
    elif np_dtype == np.float16:
        return RtFxProto.T_F16
    elif np_dtype == np.float32:
        return RtFxProto.T_F32
    elif np_dtype == np.float64:
        return RtFxProto.T_F64
    else:
        raise ValueError(
            f"Cannot convert np.dtype '{np_dtype}' to RtFxProto dtype"
        )


def np_dtype_from_rtfx_dtype(rtfx_dtype: Union[int, str]) -> np.dtype:
    """Converts an RtFx dtype to a numpy dtype."""

    if isinstance(rtfx_dtype, str):
        rtfx_dtype = RtFxProto.ElementType.Value(rtfx_dtype)

    if rtfx_dtype == RtFxProto.T_I1:
        np_dtype = bool  # Note that T_I1 data is stored as 16-bit int
    elif rtfx_dtype == RtFxProto.T_I16:
        np_dtype = np.int16
    elif rtfx_dtype == RtFxProto.T_I32:
        np_dtype = np.int32
    elif rtfx_dtype == RtFxProto.T_I64:
        np_dtype = np.int64
    elif rtfx_dtype == RtFxProto.T_U8:
        np_dtype = np.uint8
    elif rtfx_dtype == RtFxProto.T_U16:
        np_dtype = np.uint16
    elif rtfx_dtype == RtFxProto.T_U32:
        np_dtype = np.uint32
    elif rtfx_dtype == RtFxProto.T_U64:
        np_dtype = np.uint64
    elif rtfx_dtype == RtFxProto.T_F16:
        np_dtype = np.float16
    elif rtfx_dtype == RtFxProto.T_BF16:
        np_dtype = bf16
    elif rtfx_dtype == RtFxProto.T_F32:
        np_dtype = np.float32
    elif rtfx_dtype == RtFxProto.T_F64:
        np_dtype = np.float64
    else:
        raise ValueError(
            f"Cannot convert RtFxProto dtype {rtfx_dtype} to np.dtype"
        )

    return np.dtype(np_dtype)
