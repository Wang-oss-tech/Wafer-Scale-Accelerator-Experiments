# Copyright 2016-2024 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

"""
SDK utility functions
"""
import json
import struct
from typing import Dict, List, Optional

import numpy as np

MAX_MESSAGE_LENGTH = (1024 * 1024 * 1024 * 2) - 1024  # 2GB - 1 KB


def cast_uint32(x):
    """Casts a memcpy compatible type to np.uint32 which
    should be used when providing data to memcpy
    """
    if isinstance(x, (np.float16, np.int16, np.uint16)):
        z = x.view(np.uint16)
        return np.uint32(z)
    if isinstance(x, (np.float32, np.int32, np.uint32)):
        return x.view(np.uint32)
    if isinstance(x, int):
        return np.uint32(x)
    if isinstance(x, float):
        z = np.float32(x)
        return z.view(np.uint32)

    raise RuntimeError(f"type of x {type(x)} is not supported")


def memcpy_view(arr: np.array, dtype: np.dtype):
    """Returns a 32, 16 or 8 bit view of a 32 bit numpy array
    (only the lower 16 or 8 bits of each 32 bit word in the
    last two cases)

    Arguments:
    arr -- a numpy array with 4 bytes per element on which
    the numpy view will be created
    dtype -- the numpy data type which should be used in the
    output view. The itemsize must be 1, 2, or 4 bytes.
    """

    if arr.itemsize != 4:
        raise RuntimeError("memcpy_view: arr.itemsize must be 4")

    if dtype.itemsize not in [1, 2, 4]:
        raise RuntimeError(
            """memcpy_view should be used to convert to 1, 2, or 4 byte
            data types"""
        )

    view_arr = arr.view(dtype)
    odata = np.lib.stride_tricks.as_strided(
        view_arr, strides=arr.strides, shape=arr.shape
    )
    return odata


def is_valid_primitive_type(para_type: str) -> bool:
    """Check if a string represents a supported type
    """
    if para_type == 'f32':
        return True
    elif para_type == 'f16':
        return True
    elif para_type == 'i32':
        return True
    elif para_type == 'u32':
        return True
    elif para_type == 'i16':
        return True
    elif para_type == 'u16':
        return True
    else:
        return False


def parse_host_callable_api(rpc_symbols):
    """Parse host-callable API:
    api_info_dict[api name] = ( var_list, type_list )

    Example:
    api_info_dict = {'GEMV': (['alpha', 'beta', 'M', 'N'],
    ['f32', 'f32', 'i16', 'i16']), 'NRM': (['M'], ['i16'])}
    Ref: hpc_apps/test/depipelined/phase3/translate_layout.py

    fn foo(x: f32, alpha: u16, beta: i32) void {
    @export_name("foo", fn(f32, u16, i32)void);
     {
      "id": 7,
      "inputs": [
        {
          "name": "x",
          "type": "f32"
        },
        {
          "name": "alpha",
          "type": "u16"
        },
        {
          "name": "beta",
          "type": "i32"
        }
      ],
      "kind": "Func",
      "name": "foo",
      "sym_name": "foo",
      "type": "void"  --> return type
    }
    """
    api_info_dict = {}  # dictionary of host-callable functions
    for sym in rpc_symbols:
        # sym = {'id': 1, 'immutable': 0, 'kind': 'Var', 'name': 'A',
        # 'sym_name': 'ptr_A', 'type': '[*]f32'}
        # sym = {'id': 6, 'kind': 'Func', 'name': 'f_memcpy_timestamps',
        # 'sym_name': 'f_memcpy_timestamps', 'type': 'void'}
        # sym is a dict
        # fn_id = sym['id'] # integer
        kind = sym['kind']
        name = sym['name']
        # if function "name" has no arguments, the key "inputs" does not exist
        if 'inputs' in sym:
            # type(input_params) = <class 'list'>
            # input_params = [{'name': 'x', 'type': 'f32'},
            # {'name': 'alpha', 'type': 'u16'}, {'name': 'beta', 'type': 'i32'}]
            input_params = sym['inputs']
            for para in input_params:
                para_name = para['name']
                para_type = para['type']

        if kind == 'Func':
            if 'inputs' in sym:
                var_list = []
                type_list = []
                for para in input_params:
                    para_name = para['name']
                    para_type = para['type']
                    var_list.append(para_name)
                    type_list.append(para_type)
                    if not is_valid_primitive_type(para_type):
                        raise RuntimeError(
                            f"{para_type} is not a supported " "type"
                        )
                api_info_dict[name] = (var_list, type_list)
            else:
                api_info_dict[name] = ([], [])

    return api_info_dict


def get_api_info_dict(app_path: str) -> Dict:
    """Parse a compiled directory to build the host-callable
    function data dictionary

    Argument:
    app_path: The path to the compiled directory
    """

    # retrieve host-callable API
    filename = f"{app_path}/bin/out_rpc.json"
    with open(filename, "r") as f:
        dict_json = json.load(f)

    rpc_symbols = dict_json['rpc_symbols']
    api_info_dict = parse_host_callable_api(rpc_symbols)

    return api_info_dict


def get_api_info_dict_from_json(json_path: str) -> Dict:
    """Parse a compiled directory to build the host-callable
    function data dictionary

    Argument:
    json_path: The path to the out_rpc.json file
    """

    # retrieve host-callable API
    filename = f"{json_path}"
    with open(filename, "r") as f:
        dict_json = json.load(f)

    rpc_symbols = dict_json['rpc_symbols']
    api_info_dict = parse_host_callable_api(rpc_symbols)

    return api_info_dict


def check_rpc_api(name: str, arg_list: List, api_info_dict: Dict) -> bool:
    """Verify host-callable API.

    Arguments:
    name -- The name of the device function the host will call
    arg_list -- The list of arguments to the device function
    api_info_dict -- The dictionary obtained with get_api_info_dict()
    """
    (var_list, type_list) = api_info_dict[name]

    if len(var_list) != len(arg_list):
        raise RuntimeError(
            "API call requires a different number of "
            f"arguments ({len(var_list)}) from what has been "
            f"provided ({len(arg_list)})"
        )

    for arg, __type in zip(arg_list, type_list):
        if type(arg) in (float, np.float32):
            dtype = "f32"
        elif type(arg) in (np.float16,):
            dtype = "f16"
        elif type(arg) in (int,):
            if __type not in ("i32", "u32"):
                raise RuntimeError(
                    f"Error: type of {arg} should be i32 or u32")
        elif type(arg) in (np.int_, np.int32):
            dtype = "i32"
        elif type(arg) in (np.uint, np.uint32):
            dtype = "u32"
        elif type(arg) in (np.int16,):
            dtype = "i16"
        elif type(arg) in (np.uint16,):
            dtype = "u16"
        else:
            raise RuntimeError(
                f"Error [check_rpc_api]: type of {arg} is " "not supported."
            )

        if __type != dtype:
            raise RuntimeError(f"Error: type of {arg} should be {dtype}")
    return True


def input_array_to_u32(
    np_arr: np.ndarray, sentinel: Optional[int], fast_dim_sz: int
) -> np.ndarray:
    """Convert a 16-bit tensor to a 32-bit tensor of type u32. The parameter
    sentinel distiguishes the following two different extensions:
    1) zero extension
      sentinel = None
    2) upper 16-bit is the index of the array
      sentinel is Not None
    """
    assert np_arr.ndim == 1, f"array must be flat, but has shape {np_arr.shape}"

    # For 32 bit types, just reinterpret the bits
    if np_arr.itemsize == 4:
        return np_arr.view(np.uint32)

    # For 16 bit types, we pack the index of the innermost dimension into
    # the high bits of the 32 bit wavelet IF sentinels are enabled
    # (sparsity). If sentinel is None, then the high bits are zeros.
    assert np_arr.itemsize == 2, "only 16 and 32 bit input tensors supported"

    floating = np_arr.dtype == np.dtype("float16")

    new_arr = np.zeros(np_arr.shape, dtype=np.uint32)
    for (i,), el in np.ndenumerate(np_arr):
        ii = (i % fast_dim_sz) if sentinel is not None else 0
        if floating:
            el32 = cast_uint32(el)
            assert el32 <= 0xFFFF, "Must be fp16"
        else:
            el32 = np.uint32(np.uint16(el))
        new_arr[i] = np.uint32(ii << 16) | el32

    return new_arr

# Utilities for calculating cycle counts
########################################

def float_to_hex(f: np.float32) -> str:
    """Returns a hex string from a float32, used by calculate_cycles
    to convert timestamp array to human-readable elapsed cycles.

    Arguments:
    f: float32 value to convert to hex string
    """
    return hex(struct.unpack('<I', struct.pack('<f', f))[0])

def make_u48(words: np.ndarray) -> np.int64:
    """Converts the three uint16 values in words array
    into single unsigned 48-bit value, as type int64.
    Used by sub_ts to produce final timestamps.

    Arguments:
    words: numpy array of 3 uint16 values
    """
    return words[0] + (words[1] << 16) + (words[2] << 32)

def sub_ts(words: np.ndarray) -> np.int64:
    """Returns cycle count from processed timestamp words.
    Used by calculate_cycles to produce final elapsed cycles.

    Arguments:
    words: numpy array of 6 uint16 values produced from
    timestamp_buf input to calculate_cycles
    """
    return make_u48(words[3:]) - make_u48(words[0:3])

def calculate_cycles(timestamp_buf: np.ndarray) -> np.int64:
    """Converts values in timestamp_buf returned from device
    into a human-readable elapsed cycle count.

    Arguments:
    timestamp_buf: array returned from device containing
    timestamp data
    """
    hex_t0 = int(float_to_hex(timestamp_buf[0]), base=16)
    hex_t1 = int(float_to_hex(timestamp_buf[1]), base=16)
    hex_t2 = int(float_to_hex(timestamp_buf[2]), base=16)

    tsc_tensor_d2h = np.zeros(6).astype(np.uint16)
    tsc_tensor_d2h[0] = hex_t0 & 0x0000ffff
    tsc_tensor_d2h[1] = (hex_t0 >> 16) & 0x0000ffff
    tsc_tensor_d2h[2] = hex_t1 & 0x0000ffff
    tsc_tensor_d2h[3] = (hex_t1 >> 16) & 0x0000ffff
    tsc_tensor_d2h[4] = hex_t2 & 0x0000ffff
    tsc_tensor_d2h[5] = (hex_t2 >> 16) & 0x0000ffff

    return sub_ts(tsc_tensor_d2h)


def getOutputNameFromCompileOptions(option_list: List[str]) -> str:
    """getOutputNameFromCompileOptions returns the output name of a CSL
    compilation command contained in option_list

    option_list - a list of strings containing the compile options
    """
    # NOTE: how to handle multiple -o=<xxx>
    # case 1: -o=<xxx>
    # case 2: -o <xxx>
    # case 3: -o is not specified, default is "out"
    oname = None
    for j, val in enumerate(option_list):
        sub_val = val[0:3]
        if sub_val == "-o=":
            oname = val[3:None]  # case 1
        if sub_val == "-o":
            oname = option_list[j + 1]  # case 2
    if oname is None:
        oname = "out"  # case 3
    return oname
