# Copyright 2016-2024 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

"""This module provides the appliance debug_util class
"""
import struct
from typing import Tuple

import numpy as np

import cerebras.appliance.pb.sdk.sdk_appliance_pb2 as sdk_appliance_pb2

from cerebras.sdk.client.sdk_appliance_client import SdkRuntime
from cerebras.sdk.client.sdk_appliance_utils import get_artifact_id
from cerebras.appliance.errors import ApplianceUnknownError

RectangleStart = Tuple[int, int]
RectangleExtent = Tuple[int, int]
Rectangle = Tuple[RectangleStart, RectangleExtent]


def get_csl_debug_module_symbol_name(
    debug_module_key: str, symbol: str, legacy_naming_convention: bool = False
) -> str:
    """
    Retrieves the name of a symbol exported by a CSL debug module.

    CSL debug modules have two internal state variables we need to access:
    `buffer` and `buffer_index`. The names of these variables in the ELF file
    depend on a user-supplied key. For example, if the debug module is
    imported as:

        const debug_module = @import_module(
            "<debug>",
            .{ .key = "my_key",
               .buffer_size = 1000,
             }
        );

    then the imported debug module will expose symbols named
    `$$csl_debug.my_key.buffer` and `$$csl_debug.my_key.buffer_index`. This
    naming convention is an internal implementation detail of CSL's debug
    module, and is subject to change.

    Args:
        debug_module_key: The user-supplied key for the CSL debug module.
        symbol: The symbol to be retrieved.
        legacy_naming_convention: If True, use the older "legacy" naming
            convention. This option will be removed once migration to the new
            naming convention is complete.

    Returns:
        The ELF name of the requested debug module symbol.
    """
    return (
        f'{debug_module_key}.tr.{symbol}'
        if legacy_naming_convention
        else f'$$csl_debug.{debug_module_key}.{symbol}'
    )


class debug_util:
    """Loads ELF files on the Cerebras system in order to dump the symbols.

    The user does not need to export the symbols in the kernel. The debug_util
    dumps the core and looks for the symbols in the ELFs. If the symbol at Px.y
    is not found in the corresponding ELF, debug_util emits an error.

    The common error is either a wrong coordinate passing in get_symbol() or the
    coordinate is correct but the compiler removes the tensor due to some
    optimization technique. One can use readelf to check if the symbol exists or
    not. If not, the easy way is to export such symbol in the kernel to keep the
    symbol in the ELF.

    The current version only supports simulator runs.

    Args:
          artifact: a compiled artifact on the Wafer Scale Cluster.
          sdkruntime: a SdkRuntime object using the simulator.
          SdkRuntime.load() and SdkRuntime.stop() must have been called
          prior to using debug_util.

    **Example**:

          .. code-block:: python

              from cerebras.sdk.client import SdkRuntime
              from cerebras.sdk.client.debug_util import debug_util

              # run the app
              # artifact is the compiled artifact identifier
              simulator = SdkRuntime(artifact, simulator=True)
              simulator.start()
              ...
              simulator.stop()

              # retrieve symbols after the run
              debug_mod = debug_util(artifact, simulator)
              # assume the core rectangle starts at P4.1, the dimension is
              # width-by-height and we want to retrieve the symbol y for every PE
              core_offset_x = 4
              core_offset_y = 1
              for py in range(height):
                for px in range(width):
                  t = debug_mod.get_symbol(core_offset_x+px, core_offset_y+py, 'y', np.float32)
                  print(f"At (py, px) = {py, px}, symbol y = {t}")

    """

    def __init__(self, artifact: str, sdkruntime: SdkRuntime):

        self._runtime = sdkruntime
        self._app_hash = get_artifact_id(artifact)

        assert self._runtime.is_simulator(), "Only supports simulator"

    def get_symbol(
        self, col: int, row: int, symbol: str, dtype: np.dtype,
    ) -> np.ndarray:
        """
    Read the value of 'symbol' of given type at given PE coordinates.
    Note: each call to this function scans the whole fabric, so prefer
            'get_symbol_rect' over calling this in a loop.

    Args:
            col: Column of the PE
            row: Row of the PE
            symbol: Symbol to read
            dtype: Numpy dtype of values contained by symbol

    Returns:
           Numpy array of symbol values.
    """
        res = self.get_symbol_rect(((col, row), (1, 1)), symbol, dtype)
        return res[0][0]

    def get_symbol_rect(
        self, rectangle: Rectangle, symbol: str, dtype: np.dtype,
    ) -> np.ndarray:
        """
    Read the value of 'symbol' of given type for multiple PEs.

    Args:
            rectangle: Rectangle specified as ((col, row), (width, height))
            symbol: Symbol to read
            dtype: Numpy dtype of values contained by symbol

    Returns:
            Numpy array of symbol values. The first two dimensions of the
            returned array are PE coordinates (column, row) relative to the
            rectangle.
    """
        # make sure it is a true dtype
        dtype = np.dtype(dtype)
        (col, row), (width, height) = rectangle

        request = sdk_appliance_pb2.SdkGetSymbolParams(
            app_hash=self._app_hash,
            x=col,
            y=row,
            w=width,
            h=height,
            symbol=symbol,
            bytes_per_elem=dtype.itemsize,
        )

        response_iterator = self._runtime.stub().sdk_get_symbol_rect(request)

        res = None
        offset = 0
        for response in response_iterator:
            if response.HasField("status"):
                raise ApplianceUnknownError(response.status.message)
            else:
                if res is None:
                    num_bytes = response.d2h_data.total_bytes
                    bytes_per_pe = int(num_bytes / (width * height))
                    elem_per_pe = int(bytes_per_pe / dtype.itemsize)
                    res = np.empty((width, height, elem_per_pe), dtype=dtype)

                data = response.d2h_data
                num_elems = int(data.num_bytes / dtype.itemsize)
                res.ravel()[offset: offset + num_elems] = np.frombuffer(
                    data.data_chunk, dtype=dtype
                )
                offset += num_elems

        return res

    def read_trace(self, col: int, row: int, key: str) -> list:
        """Parse a CSL trace buffer.

        Args:
            col: Column of PE that contains trace buffer
            row: Row of PE that contains trace buffer
            key:
                Key of imported trace module. For instance, if the CSL code
                contains:

                    const t = @import_module(
                      "<debug>",
                      .{ .key = "my_trace",
                         .buffer_size = ...
                       }
                    );

                Then 'key' is 'my_trace'.

        Returns:
            Heterogeneous list of trace values.
        """

        def get_buffer_index(legacy: bool):
            sym = get_csl_debug_module_symbol_name(
                key, 'buffer_index', legacy_naming_convention=legacy
            )
            return self.get_symbol(col, row, sym, np.uint16)[0]

        try:
            buffer_index = get_buffer_index(legacy=False)
        except RuntimeError:
            buffer_index = get_buffer_index(legacy=True)

        def get_buff(legacy: bool):
            sym = get_csl_debug_module_symbol_name(
                key, 'buffer', legacy_naming_convention=legacy
            )
            return self.get_symbol(col, row, sym, np.uint8)

        try:
            buff = get_buff(legacy=False)
        except RuntimeError:
            buff = get_buff(legacy=True)

        # buffer_index is the index into the buffer array, which is an array of u16
        # values. We fetch the buffer as np.uint8 (bytes), so we need to multiply
        # the index by 2 to translate from 16-bit words to bytes.
        buffer_index *= 2
        buff = list(buff[:buffer_index])
        result = []
        while buff:
            # The tag on device is 16 bits, but the type tag is always the lower 8
            # bits. Recall that device is little-endian.
            tag = chr(buff.pop(0))
            tag_high_bits = buff.pop(0)

            def _unpack(tag: str, packed: bytes):
                # 't' is the tag used by the CSL trace library for 'timestamp',
                # and does not exist in the 'struct' library
                if tag == 't':
                    # A timestamp is 48 bits, which is not supported by 'struct'
                    # so pad with zeros and parse as a 64-bit int
                    packed = b''.join([packed, b'\x00\x00'])
                    tag = 'Q'
                result = struct.unpack(f'<{tag}', packed)[0]
                # if the tag is terminated with 's', we need to decode the bytes
                # into a python string
                if tag[-1] == 's':
                    result = result.split(b'\x00')[0].decode('utf-8')
                return result

            if tag in '?bB':  # bools, i8, and u8, respectively
                # values are single-byte and the trace library packs them into the
                # high bits of the tag_word
                result.append(_unpack(tag, tag_high_bits))
                continue

            # 't' is the tag used by the CSL trace library for 'timestamp', and does
            # not exist in the 'struct' library
            if tag == 't':
                size = 6  # a timestamp is 48 bits, so 6 bytes
            # 's' is the tag use by a utf-8 string, and we pack the number of bytes
            # in the buffer in the second byte. The buffer size is always even.
            # The maximum string size is the maximum even buffer length (254).
            elif tag == 's':
                size = tag_high_bits
                tag = str(size) + tag
            else:
                size = struct.Struct(f'<{tag}').size

            assert size % 2 == 0

            packed = b''.join(buff[:size])
            buff = buff[size:]
            result.append(_unpack(tag, packed))

        return result
