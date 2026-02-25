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

from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime, MemcpyDataType, MemcpyOrder # pylint: disable=no-name-in-module

# Read arguments
parser = argparse.ArgumentParser()
parser.add_argument('--name', help="the test compile output dir")
parser.add_argument('--cmaddr', help="IP:port for CS system")
args = parser.parse_args()

# Get matrix dimensions from compile metadata
with open(f"{args.name}/out.json", encoding='utf-8') as json_file:
  compile_data = json.load(json_file)

# Matrix dimensions
N = int(compile_data['params']['N']) # columns
M = int(compile_data['params']['M']) # rows

# PE grid dimensions
kernel_x_dim = int(compile_data['params']['kernel_x_dim'])
kernel_y_dim = int(compile_data['params']['kernel_y_dim'])

# Colors used for memcpy streaming
MEMCPYH2D_DATA_1 = int(compile_data['params']['MEMCPYH2D_DATA_1_ID'])
MEMCPYH2D_DATA_2 = int(compile_data['params']['MEMCPYH2D_DATA_2_ID'])
MEMCPYH2D_DATA_3 = int(compile_data['params']['MEMCPYH2D_DATA_3_ID'])
MEMCPYD2H_DATA_1 = int(compile_data['params']['MEMCPYD2H_DATA_1_ID'])

# Construct A, x, b
A = np.arange(M*N, dtype=np.float32).reshape(M,N)
x = np.full(shape=N, fill_value=1.0, dtype=np.float32)
b = np.full(shape=M, fill_value=2.0, dtype=np.float32)

# Calculate expected y
y_expected = A@x + b

# Size of N dimension on each PE
N_per_PE = N // kernel_x_dim
M_per_PE = M // kernel_y_dim

# Construct a runner using SdkRuntime
runner = SdkRuntime(args.name, cmaddr=args.cmaddr)

# Load and run the program
runner.load()
runner.run()

# Prepare A for streaming: each PE gets its chunk stored column major
A_prepared = A.reshape(kernel_y_dim, M_per_PE, kernel_x_dim, N_per_PE).transpose(0, 2, 3, 1).ravel()

# OVERLAP STRATEGY - MAXIMUM CONCURRENCY:
# 1. Stream A, x, and b ALL concurrently (all use nonblock=True)
# 2. Device starts computing as soon as A is ready and x elements arrive
# 3. Device computes Ax while A/x/b are still streaming (maximum overlap!)
# 4. Synchronization happens when reading y back (memcpy_d2h with nonblock=False)

# Stream A into all PEs - DON'T WAIT (nonblock=True)
runner.memcpy_h2d(MEMCPYH2D_DATA_3, A_prepared, 0, 0, kernel_x_dim, kernel_y_dim, M_per_PE*N_per_PE,
  streaming=True, order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=True)

# Stream x into top row - DON'T WAIT (nonblock=True)
# Device will start computing Ax as x elements arrive (while A/x still streaming)
runner.memcpy_h2d(MEMCPYH2D_DATA_1, x, 0, 0, kernel_x_dim, 1, N_per_PE, streaming=True,
  order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=True)

# Stream b into left column - DON'T WAIT (nonblock=True)
# Maximum host-side concurrency: all three inputs stream in parallel!
# Device computes Ax while b is streaming (overlap achieved)
runner.memcpy_h2d(MEMCPYH2D_DATA_2, b, 0, 0, 1, kernel_y_dim, M_per_PE, streaming=True,
  order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=True)

# Stream y back from PEs (kernel_x_dim-1, 0) and (kernel_x_dim-1, kernel_y_dim-1)
# SYNCHRONIZATION POINT: nonblock=False ensures all previous operations complete:
# - All streaming (A, x, b) finishes
# - Device computation (Ax + b) finishes
# - Result is ready before we proceed
y_result = np.zeros([M], dtype=np.float32)
runner.memcpy_d2h(y_result, MEMCPYD2H_DATA_1, kernel_x_dim-1, 0, 1, kernel_y_dim, M_per_PE,
  streaming=True, order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT,
  nonblock=False)

# Stop the program
runner.stop()

# Ensure that the result matches our expectation
np.testing.assert_allclose(y_result, y_expected)
print("SUCCESS!")
