#!/usr/bin/env cs_python

# SUMMA Matrix Multiplication with Pipelined Broadcasts
#
# Host program to:
# 1. Generate random A and B matrices
# 2. Distribute tiles to PEs
# 3. Run pipelined SUMMA kernel
# 4. Collect and verify results

import argparse
import json
import numpy as np

from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime
from cerebras.sdk.runtime.sdkruntimepybind import MemcpyDataType
from cerebras.sdk.runtime.sdkruntimepybind import MemcpyOrder

parser = argparse.ArgumentParser()
parser.add_argument("--name", help="the test name")
parser.add_argument("--cmaddr", help="IP:port for CS system")
args = parser.parse_args()

# Get params from compile metadata
with open(f"{args.name}/out.json", encoding='utf-8') as json_file:
    compile_data = json.load(json_file)

# Kernel rectangle and per-PE matrix dimensions
P = int(compile_data['params']['P'])
Mt = int(compile_data['params']['Mt'])
Kt = int(compile_data['params']['Kt'])
Nt = int(compile_data['params']['Nt'])

# Full matrix dimensions
M = Mt * P
K = Kt * P
N = Nt * P

print(f"SUMMA with Manual Multicasting")
print(f"  Grid: {P} x {P} PEs")
print(f"  Tile sizes: Mt={Mt}, Kt={Kt}, Nt={Nt}")
print(f"  Full matrices: A({M}x{K}) @ B({K}x{N}) = C({M}x{N})")

memcpy_dtype = MemcpyDataType.MEMCPY_32BIT
memcpy_order = MemcpyOrder.ROW_MAJOR

# Use a deterministic seed
np.random.seed(seed=7)

A = np.random.rand(M, K).astype(np.float32)
B = np.random.rand(K, N).astype(np.float32)

runner = SdkRuntime(args.name, cmaddr=args.cmaddr)

sym_A = runner.get_id("A")
sym_B = runner.get_id("B")
sym_C = runner.get_id("C")

runner.load()
runner.run()

w = P
h = P

# Transform matrices to column-major tile distribution
print("Copying A to device...")
A1 = A.reshape(h, Mt, w, Kt)
A2 = A1.transpose(0, 2, 3, 1)
A3 = A2.reshape(h, w, Mt*Kt)
runner.memcpy_h2d(sym_A, A3.ravel(), 0, 0, w, h, Mt*Kt,
    streaming=False, data_type=memcpy_dtype, order=MemcpyOrder.ROW_MAJOR, nonblock=True)

print("Copying B to device...")
B1 = B.reshape(h, Kt, w, Nt)
B2 = B1.transpose(0, 2, 3, 1)
B3 = B2.reshape(h, w, Kt*Nt)
runner.memcpy_h2d(sym_B, B3.ravel(), 0, 0, w, h, Kt*Nt,
    streaming=False, data_type=memcpy_dtype, order=MemcpyOrder.ROW_MAJOR, nonblock=True)

print("Running pipelined SUMMA kernel...")
runner.launch("main", nonblock=False)

print("Copying C from device...")
C3_1d_u32 = np.zeros(h*w*Mt*Nt, np.uint32)
runner.memcpy_d2h(C3_1d_u32, sym_C, 0, 0, w, h, Mt*Nt,
    streaming=False, data_type=memcpy_dtype, order=MemcpyOrder.ROW_MAJOR, nonblock=False)

# Transform back from tile distribution
C3 = C3_1d_u32.reshape((h, w, Nt, Mt))
C2 = C3.transpose(0, 3, 1, 2)
C1 = C2.reshape(M, N)
C = C1.view(np.float32)

runner.stop()

# Verify
print("Verifying results...")
C_expected = np.dot(A, B)
np.testing.assert_allclose(C_expected, C, rtol=1e-05, atol=1e-06)

print("SUCCESS")
