#!/usr/bin/env cs_python

# 2D SUMMA Test
# Validates distributed matrix multiplication: C = A @ B

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

P = int(compile_data['params']['P'])
Mt = int(compile_data['params']['Mt'])
Kt = int(compile_data['params']['Kt'])
Nt = int(compile_data['params']['Nt'])

# Full matrix dimensions
M = Mt * P
K = Kt * P
N = Nt * P

print(f"Running 2D SUMMA test:")
print(f"  Grid: {P}x{P} PEs")
print(f"  Tile sizes: Mt={Mt}, Kt={Kt}, Nt={Nt}")
print(f"  Full matrices: A({M}x{K}) @ B({K}x{N}) = C({M}x{N})")

memcpy_dtype = MemcpyDataType.MEMCPY_32BIT
memcpy_order = MemcpyOrder.ROW_MAJOR

# Use deterministic seed for reproducibility
np.random.seed(seed=7)

# Generate random input matrices
A = np.random.rand(M, K).astype(np.float32)
B = np.random.rand(K, N).astype(np.float32)

print(f"  A sample: {A[0,:4]}...")
print(f"  B sample: {B[0,:4]}...")

runner = SdkRuntime(args.name, cmaddr=args.cmaddr)

sym_A = runner.get_id("A")
sym_B = runner.get_id("B")
sym_C = runner.get_id("C")

runner.load()
runner.run()

w = P  # width of PE grid
h = P  # height of PE grid

# Transform matrices for distribution across PEs
# Each PE gets a tile, stored in column-major order
#
# For A (M x K), PE at (px, py) gets A_tile = A[py*Mt:(py+1)*Mt, px*Kt:(px+1)*Kt]
# We need to reshape and transpose to get column-major tiles distributed correctly

# A transformation: (M, K) -> (h, Mt, w, Kt) -> (h, w, Kt, Mt) -> (h, w, Mt*Kt)
A1 = A.reshape(h, Mt, w, Kt)
A2 = A1.transpose(0, 2, 3, 1)  # (h, w, Kt, Mt) - makes local tile column-major
A3 = A2.reshape(h, w, Mt * Kt)

print("Copying A to device...")
runner.memcpy_h2d(sym_A, A3.ravel(), 0, 0, w, h, Mt * Kt,
    streaming=False, data_type=memcpy_dtype, order=MemcpyOrder.ROW_MAJOR, nonblock=True)

# B transformation: (K, N) -> (h, Kt, w, Nt) -> (h, w, Nt, Kt) -> (h, w, Kt*Nt)
B1 = B.reshape(h, Kt, w, Nt)
B2 = B1.transpose(0, 2, 3, 1)  # (h, w, Nt, Kt) - makes local tile column-major
B3 = B2.reshape(h, w, Kt * Nt)

print("Copying B to device...")
runner.memcpy_h2d(sym_B, B3.ravel(), 0, 0, w, h, Kt * Nt,
    streaming=False, data_type=memcpy_dtype, order=MemcpyOrder.ROW_MAJOR, nonblock=False)

print("Launching SUMMA computation...")
runner.launch("main", nonblock=False)

print("Reading C from device...")
C3_1d_u32 = np.zeros(h * w * Mt * Nt, np.uint32)
runner.memcpy_d2h(C3_1d_u32, sym_C, 0, 0, w, h, Mt * Nt,
    streaming=False, data_type=memcpy_dtype, order=MemcpyOrder.ROW_MAJOR, nonblock=False)

# C transformation: reverse of A/B transformation
# C3 is (h, w, Mt*Nt) with column-major local tiles
C3 = C3_1d_u32.reshape((h, w, Nt, Mt))  # (h, w, Nt, Mt) - column-major tile
C2 = C3.transpose(0, 3, 1, 2)  # (h, Mt, w, Nt)
C1 = C2.reshape(M, N)
C = C1.view(np.float32)

runner.stop()

# Compute expected result
C_expected = np.dot(A, B)

print(f"\nExpected C sample: {C_expected[0,:4]}...")
print(f"Computed C sample: {C[0,:4]}...")

# Validate
max_diff = np.max(np.abs(C_expected - C))
print(f"\nMax absolute difference: {max_diff}")

try:
    np.testing.assert_allclose(C_expected, C, rtol=1e-05, atol=1e-06)
    print("\nSUCCESS: SUMMA result matches expected!")
except AssertionError as e:
    print(f"\nFAILED: Results don't match!")
    print(e)

    # Debug: show some differences
    diff = np.abs(C_expected - C)
    worst_idx = np.unravel_index(np.argmax(diff), diff.shape)
    print(f"Worst difference at {worst_idx}: expected={C_expected[worst_idx]}, got={C[worst_idx]}")
    exit(1)
