#!/usr/bin/env cs_python

# Copyright 2025 Cerebras Systems.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import json
import numpy as np

from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime     # pylint: disable=no-name-in-module
from cerebras.sdk.runtime.sdkruntimepybind import MemcpyDataType # pylint: disable=no-name-in-module
from cerebras.sdk.runtime.sdkruntimepybind import MemcpyOrder    # pylint: disable=no-name-in-module
from cerebras.sdk.sdk_utils import memcpy_view                   # pylint: disable=no-name-in-module

parser = argparse.ArgumentParser()
parser.add_argument("--name", help="the test name")
parser.add_argument("--cmaddr", help="IP:port for CS system")
parser.add_argument("--warmup", default=10, type=int, help="Warmup runs (default 10)")
parser.add_argument("--repeats", default=100, type=int, help="Timed runs (default 100)")
args = parser.parse_args()

# Get params from compile metadata
with open(f"{args.name}/out.json", encoding='utf-8') as json_file:
  compile_data = json.load(json_file)

# Kernel rectangle and GEMV dimensions
kernel_rows = int(compile_data["params"]["kernel_rows"])
kernel_cols = int(compile_data["params"]["kernel_cols"])
matrix_rows = int(compile_data["params"]["matrix_rows"])
matrix_cols = int(compile_data["params"]["matrix_cols"])

# Use the same host partitioning as waferllm-gemv:
#   X is split by PE row and replicated across PE columns
#   W is tiled into (py, px) blocks of shape Mt x Nt
np.random.seed(seed=7)
X = np.random.rand(1, matrix_rows).astype(np.float16)
tensor_X = np.tile(X.reshape(kernel_rows, matrix_rows // kernel_rows), reps=(1, kernel_cols))
W = np.random.rand(matrix_rows, matrix_cols).astype(np.float16)
y_expected = (X.astype(np.float32) @ W.astype(np.float32)).astype(np.float16).reshape(matrix_cols)

# Specify path to ELF files, set up runner
runner = SdkRuntime(args.name, cmaddr=args.cmaddr)

memcpy_dtype_16 = MemcpyDataType.MEMCPY_16BIT
memcpy_dtype_32 = MemcpyDataType.MEMCPY_32BIT
memcpy_order = MemcpyOrder.ROW_MAJOR

symbol_X = runner.get_id("X")
symbol_W = runner.get_id("W")
symbol_res = runner.get_id("res")
symbol_time_memcpy = runner.get_id("time_memcpy")
symbol_time_ref = runner.get_id("time_ref")

runner.load()
runner.run()

print("Copying data...")
Mt = matrix_rows // kernel_rows
Nt = matrix_cols // kernel_cols

X_u32 = np.uint32(tensor_X.ravel().view(np.uint16))
runner.memcpy_h2d(symbol_X, X_u32, 0, 0, kernel_cols, kernel_rows, Mt,
                  streaming=False, data_type=memcpy_dtype_16, nonblock=False,
                  order=memcpy_order)

W1 = W.reshape(kernel_rows, Mt, kernel_cols, Nt)
W2 = W1.transpose(0, 2, 1, 3)
W3 = W2.reshape(kernel_rows, kernel_cols, Mt * Nt)
W_u32 = np.uint32(W3.ravel().view(np.uint16))
runner.memcpy_h2d(symbol_W, W_u32, 0, 0, kernel_cols, kernel_rows, Mt * Nt,
                  streaming=False, data_type=memcpy_dtype_16, nonblock=False,
                  order=memcpy_order)

print("Launching kernel...")
runner.launch("init_task", nonblock=False)
runner.launch("pipeline_allreduce_host", np.int16(args.warmup), np.int16(args.repeats), nonblock=False)

# Collect the replicated full-output rows, matching waferllm-gemv.
res_u32 = np.zeros(kernel_rows * matrix_cols, dtype=np.uint32)
runner.memcpy_d2h(res_u32, symbol_res, 0, 0, kernel_cols, kernel_rows, Nt,
                  streaming=False, data_type=memcpy_dtype_16, nonblock=False,
                  order=memcpy_order)
res = memcpy_view(res_u32, np.dtype(np.float16)).reshape(kernel_rows, matrix_cols)

time_memcpy = np.zeros(kernel_rows * kernel_cols * 3, dtype=np.float32)
runner.memcpy_d2h(time_memcpy, symbol_time_memcpy, 0, 0, kernel_cols, kernel_rows, 3,
                  streaming=False, data_type=memcpy_dtype_32, nonblock=False,
                  order=memcpy_order)
time_ref = np.zeros(kernel_rows * kernel_cols * 2, dtype=np.float32)
runner.memcpy_d2h(time_ref, symbol_time_ref, 0, 0, kernel_cols, kernel_rows, 2,
                  streaming=False, data_type=memcpy_dtype_32, nonblock=False,
                  order=memcpy_order)
runner.stop()
print("Copied back result.")

print("res calculated (row 0): ", res[0])
print("y expected:             ", y_expected)
expected_rows = np.tile(y_expected.reshape(1, matrix_cols), reps=(kernel_rows, 1))
np.testing.assert_allclose(res, expected_rows, rtol=0.5, atol=0)
print("SUCCESS")
