#!/usr/bin/env cs_python

# SUMMA Matrix Multiplication with Manual Multicasting
#
# Host program to:
# 1. Generate random A and B matrices
# 2. Distribute tiles to PEs
# 3. Run SUMMA kernel
# 4. Collect and verify results
# 5. Read cycle-count timestamps and print performance

import argparse
import json
import struct
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
# A is M x K, B is K x N, C is M x N
M = Mt * P
K = Kt * P
N = Nt * P

print(f"SUMMA with Manual Multicasting")
print(f"  Grid: {P} x {P} PEs")
print(f"  Tile sizes: Mt={Mt}, Kt={Kt}, Nt={Nt}")
print(f"  Full matrices: A({M}x{K}) @ B({K}x{N}) = C({M}x{N})")

memcpy_dtype = MemcpyDataType.MEMCPY_32BIT
memcpy_order = MemcpyOrder.ROW_MAJOR


def float_to_hex(f):
    return hex(struct.unpack("<I", struct.pack("<f", f))[0])


def make_u48(words):
    return int(words[0]) + (int(words[1]) << 16) + (int(words[2]) << 32)


# Use a deterministic seed so that CI results are predictable
np.random.seed(seed=7)

A = np.random.rand(M, K).astype(np.float32)
B = np.random.rand(K, N).astype(np.float32)

runner = SdkRuntime(args.name, cmaddr=args.cmaddr)

sym_A = runner.get_id("A")
sym_B = runner.get_id("B")
sym_C = runner.get_id("C")
sym_time_memcpy = runner.get_id("time_memcpy")
sym_time_ref = runner.get_id("time_ref")

runner.load()
runner.run()

try:
    w = P  # number of columns PEs in the core rectangle
    h = P  # number of row PEs in the core rectangle

    # How to transform a 2-D tensor into a cliff distribution with
    # column-major local tensor
    #
    # Example: w=2, h=2, A is 4-by-4 (lh-by-lw)
    # A = |  0  1  2  3 |
    #     |  4  5  6  7 |
    #     |  8  9 10 11 |
    #     | 12 13 14 15 |
    # A1 = A.reshape(2,2,2,2) of the form (h,lh,w,lw)
    # A1 = | | 0  1|  | 4  5| |
    #      | | 2  3|, | 6  7| |
    #      |                  |
    #      | | 8  9|  |12 13| |
    #      | |10 11|, |14 15| |
    # A2 = A1.transpose(0, 2, 3, 1) of the form (h, w, lw, lh)
    # so the local tensor lh-by-lw is col-major
    # A2 = | | 0  4|  | 2  6| |
    #      | | 1  5|, | 3  7| |
    #      |                  |
    #      | | 8 12|  |10 14| |
    #      | | 9 13|, |11 15| |
    # A3 = A2.reshape(2,2,4)
    # A3 = |  0  4  1  5 |
    #      |  2  6  3  7 |
    #      |  8 12  9 13 |
    #      | 10 14 11 15 |
    # A3 is h-w-l

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

    print("Running SUMMA kernel...")
    runner.launch("main", nonblock=False)

    # Copy cycle-count timestamps from device (written in exit_task)
    time_memcpy_1d_f32 = np.zeros(P * P * 3, dtype=np.float32)
    runner.memcpy_d2h(
        time_memcpy_1d_f32, sym_time_memcpy, 0, 0, P, P, 3,
        streaming=False, data_type=MemcpyDataType.MEMCPY_32BIT,
        order=MemcpyOrder.ROW_MAJOR, nonblock=False
    )
    time_memcpy_hwl = np.reshape(time_memcpy_1d_f32, (P, P, 3), order="C")

    time_ref_1d_f32 = np.zeros(P * P * 2, dtype=np.float32)
    runner.memcpy_d2h(
        time_ref_1d_f32, sym_time_ref, 0, 0, P, P, 2,
        streaming=False, data_type=MemcpyDataType.MEMCPY_32BIT,
        order=MemcpyOrder.ROW_MAJOR, nonblock=False
    )
    time_ref_hwl = np.reshape(time_ref_1d_f32, (P, P, 2), order="C")

    # Unpack 48-bit cycle counts from f32 bit patterns (use ii,jj to avoid clobbering w,h)
    time_start = np.zeros((P, P), dtype=np.int64)
    time_end = np.zeros((P, P), dtype=np.int64)
    word = np.zeros(3, dtype=np.uint16)
    for jj in range(P):
        for ii in range(P):
            hex_t0 = int(float_to_hex(time_memcpy_hwl[ii, jj, 0]), base=16)
            hex_t1 = int(float_to_hex(time_memcpy_hwl[ii, jj, 1]), base=16)
            hex_t2 = int(float_to_hex(time_memcpy_hwl[ii, jj, 2]), base=16)
            word[0] = hex_t0 & 0x0000FFFF
            word[1] = (hex_t0 >> 16) & 0x0000FFFF
            word[2] = hex_t1 & 0x0000FFFF
            time_start[ii, jj] = make_u48(word)
            word[0] = (hex_t1 >> 16) & 0x0000FFFF
            word[1] = hex_t2 & 0x0000FFFF
            word[2] = (hex_t2 >> 16) & 0x0000FFFF
            time_end[ii, jj] = make_u48(word)

    time_ref = np.zeros((P, P), dtype=np.int64)
    for jj in range(P):
        for ii in range(P):
            hex_t0 = int(float_to_hex(time_ref_hwl[ii, jj, 0]), base=16)
            hex_t1 = int(float_to_hex(time_ref_hwl[ii, jj, 1]), base=16)
            word[0] = hex_t0 & 0x0000FFFF
            word[1] = (hex_t0 >> 16) & 0x0000FFFF
            word[2] = hex_t1 & 0x0000FFFF
            time_ref[ii, jj] = make_u48(word)

    for py in range(P):
        for px in range(P):
            time_ref[py, px] -= px + py
    time_start = time_start - time_ref
    time_end = time_end - time_ref

    min_time_start = time_start.min()
    max_time_end = time_end.max()
    print(f"  Mean cycle count: {np.mean(time_end - time_start):.1f}")
    print(f"  Max cycle count:  {max_time_end - min_time_start}")

    print("Copying C from device...")
    expected_c_size = M * N  # P*P*Mt*Nt
    C3_1d_u32 = np.zeros(expected_c_size, np.uint32)
    runner.memcpy_d2h(C3_1d_u32, sym_C, 0, 0, P, P, Mt*Nt,
        streaming=False, data_type=memcpy_dtype, order=MemcpyOrder.ROW_MAJOR, nonblock=False)

    actual_c_size = C3_1d_u32.size
    if actual_c_size != expected_c_size:
        raise ValueError(
            f"C size mismatch: device returned {actual_c_size} elements, "
            f"expected M*N = {expected_c_size} (P={P}, Mt={Mt}, Nt={Nt}). "
            "Ensure the binary in out/ was compiled with the same params as out.json; "
            "run ./commands_wse2.sh to recompile and run with consistent params."
        )

    # C3 is h-by-w-l or
    # C3 is of the form (P, P, Nt, Mt) where local tensor Mt-by-Nt is column-major
    C3 = C3_1d_u32.reshape((P, P, Nt, Mt))
    # C2 is of the form (h, Mt, w, Nt)
    C2 = C3.transpose(0, 3, 1, 2)
    # C1 is of the form (M, N)
    C1 = C2.reshape(M, N)
    # C has the correct data type
    C = C1.view(np.float32)

    # Check the result
    print("Verifying results...")
    C_expected = np.dot(A, B)

    # absolute(a - b) <= (atol + rtol * absolute(b))
    np.testing.assert_allclose(C_expected, C, rtol=1e-05, atol=1e-06)

    print("SUCCESS")
finally:
    runner.stop()
