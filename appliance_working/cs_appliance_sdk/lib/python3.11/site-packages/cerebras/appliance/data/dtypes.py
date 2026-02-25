# Copyright 2016-2023 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

"""Custom numpy dtypes.

Attributes:
    bf16: A custom BFloat16 dtype view. This is a 2-byte integer dtype that
        holds data that corresponds to a BFloat16 value. Note that this dtype
        is just a data holder and should not be used for any arithmetic or
        direct conversions to other dtypes. Instead, construct a real BFloat16
        array from this dtype using `np.view` method to reinterpret the data
        as a BFloat16 array. In order to check if a given array or dtype is
        `bf16`, do NOT use `dtype == bf16`. Instead, use `is_bf16` method
        defined in this module.
"""

from typing import Union

import numpy as np

# A custom BFloat16 dtype view.
bf16 = np.dtype("i2", metadata={"is_bfloat16": True, "is_cerebras": True})


def is_bf16(dtype_or_array: Union[np.ndarray, np.dtype], /) -> bool:
    """Returns True if the given array or dtype has/is a `bf16` dtype.

    Args:
        dtype_or_array: The array or dtype to check.
    Returns:
        True if the given array or dtype is `bf16`.
    """
    if isinstance(dtype_or_array, np.ndarray):
        dtype_or_array = dtype_or_array.dtype

    return bool(
        isinstance(dtype_or_array, np.dtype)
        and (dtype_or_array == np.int16 or dtype_or_array == np.uint16)
        and dtype_or_array.metadata is not None
        and all(
            dtype_or_array.metadata.get(k) == v
            for k, v in bf16.metadata.items()
        )
    )
