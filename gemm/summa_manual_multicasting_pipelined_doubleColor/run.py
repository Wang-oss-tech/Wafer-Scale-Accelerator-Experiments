# #!/usr/bin/env cs_python

# # SUMMA Matrix Multiplication with 4-Color Overlapped Broadcasts
# #
# # Two-pass execution (matching gemm-collectives_2d-fp16 structure):
# #   Pass 1: summa_host(0, 1)  — correctness check
# #   Pass 2: summa_host(5, 50) — performance timing (commented out)

# import argparse
# import json
# import struct
# import numpy as np

# from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime
# from cerebras.sdk.runtime.sdkruntimepybind import MemcpyDataType
# from cerebras.sdk.runtime.sdkruntimepybind import MemcpyOrder


# def float_to_hex(f):
#     return hex(struct.unpack("<I", struct.pack("<f", f))[0])


# def make_u48(words):
#     return int(words[0]) + (int(words[1]) << 16) + (int(words[2]) << 32)


# parser = argparse.ArgumentParser()
# parser.add_argument("--name",   required=True, help="the test name")
# parser.add_argument("--cmaddr", default=None,  help="IP:port for CS system")
# args = parser.parse_args()

# with open(f"{args.name}/out.json", encoding='utf-8') as json_file:
#     compile_data = json.load(json_file)

# P  = int(compile_data['params']['P'])
# Mt = int(compile_data['params']['Mt'])
# Kt = int(compile_data['params']['Kt'])
# Nt = int(compile_data['params']['Nt'])

# M = Mt * P
# K = Kt * P
# N = Nt * P

# print(f"SUMMA with 4-Color Overlapped Broadcasts")
# print(f"  Grid: {P} x {P} PEs")
# print(f"  Tile sizes: Mt={Mt}, Kt={Kt}, Nt={Nt}")
# print(f"  Full matrices: A({M}x{K}) @ B({K}x{N}) = C({M}x{N})")

# memcpy_dtype = MemcpyDataType.MEMCPY_32BIT
# memcpy_order = MemcpyOrder.ROW_MAJOR

# np.random.seed(seed=7)
# A = np.random.rand(M, K).astype(np.float32)
# B = np.random.rand(K, N).astype(np.float32)

# # Cliff distribution — column-major local tiles
# A1 = A.reshape(P, Mt, P, Kt)
# A2 = A1.transpose(0, 2, 3, 1)
# A3 = A2.reshape(P, P, Mt * Kt)

# B1 = B.reshape(P, Kt, P, Nt)
# B2 = B1.transpose(0, 2, 3, 1)
# B3 = B2.reshape(P, P, Kt * Nt)

# runner = SdkRuntime(args.name, cmaddr=args.cmaddr)

# sym_A           = runner.get_id("A")
# sym_B           = runner.get_id("B")
# sym_C           = runner.get_id("C")
# sym_time_memcpy = runner.get_id("time_memcpy")
# sym_time_ref    = runner.get_id("time_ref")

# runner.load()
# runner.run()

# try:
#     # Send input tiles (once; tiles stay on device across both passes)
#     print("Copying A to device...")
#     runner.memcpy_h2d(sym_A, A3.ravel(), 0, 0, P, P, Mt * Kt,
#                       streaming=False, data_type=memcpy_dtype,
#                       order=memcpy_order, nonblock=False)

#     print("Copying B to device...")
#     runner.memcpy_h2d(sym_B, B3.ravel(), 0, 0, P, P, Kt * Nt,
#                       streaming=False, data_type=memcpy_dtype,
#                       order=memcpy_order, nonblock=False)

#     # --- Pass 1: correctness check (0 warmup, 1 repeat) ---
#     total_warmup_times, total_repeat_times = 0, 1
#     print("Running SUMMA kernel (correctness check)...")
#     runner.launch("summa_host", np.int16(total_warmup_times),
#                   np.int16(total_repeat_times), nonblock=False)

#     C3_1d_u32 = np.zeros(P * P * Mt * Nt, dtype=np.uint32)
#     runner.memcpy_d2h(C3_1d_u32, sym_C, 0, 0, P, P, Mt * Nt,
#                       streaming=False, data_type=memcpy_dtype,
#                       order=memcpy_order, nonblock=False)

#     # --- Pass 2: performance timing (5 warmup, 50 repeats) ---
#     # total_warmup_times, total_repeat_times = 5, 50
#     # runner.launch("summa_host", np.int16(total_warmup_times),
#     #               np.int16(total_repeat_times), nonblock=False)

#     # Retrieve timestamps
#     time_memcpy_1d_f32 = np.zeros(P * P * 3, dtype=np.float32)
#     runner.memcpy_d2h(time_memcpy_1d_f32, sym_time_memcpy, 0, 0, P, P, 3,
#                       streaming=False, data_type=MemcpyDataType.MEMCPY_32BIT,
#                       order=MemcpyOrder.ROW_MAJOR, nonblock=False)
#     time_memcpy_hwl = np.reshape(time_memcpy_1d_f32, (P, P, 3), order="C")

#     time_ref_1d_f32 = np.zeros(P * P * 2, dtype=np.float32)
#     runner.memcpy_d2h(time_ref_1d_f32, sym_time_ref, 0, 0, P, P, 2,
#                       streaming=False, data_type=MemcpyDataType.MEMCPY_32BIT,
#                       order=MemcpyOrder.ROW_MAJOR, nonblock=False)
#     time_ref_hwl = np.reshape(time_ref_1d_f32, (P, P, 2), order="C")

# finally:
#     runner.stop()

# # Correctness check
# C3 = C3_1d_u32.reshape((P, P, Nt, Mt))
# C2 = C3.transpose(0, 3, 1, 2)
# C1 = C2.reshape(M, N)
# C  = C1.view(np.float32)

# print("Verifying results...")
# C_expected = np.dot(A, B)
# np.testing.assert_allclose(C_expected, C, rtol=1e-05, atol=1e-06)
# print("Correctness check PASSED")

# # Unpack 48-bit cycle counts
# time_start = np.zeros((P, P), dtype=np.int64)
# time_end   = np.zeros((P, P), dtype=np.int64)
# word = np.zeros(3, dtype=np.uint16)
# for w in range(P):
#     for h in range(P):
#         hex_t0 = int(float_to_hex(time_memcpy_hwl[h, w, 0]), base=16)
#         hex_t1 = int(float_to_hex(time_memcpy_hwl[h, w, 1]), base=16)
#         hex_t2 = int(float_to_hex(time_memcpy_hwl[h, w, 2]), base=16)
#         word[0] = hex_t0 & 0x0000ffff
#         word[1] = (hex_t0 >> 16) & 0x0000ffff
#         word[2] = hex_t1 & 0x0000ffff
#         time_start[h, w] = make_u48(word)
#         word[0] = (hex_t1 >> 16) & 0x0000ffff
#         word[1] = hex_t2 & 0x0000ffff
#         word[2] = (hex_t2 >> 16) & 0x0000ffff
#         time_end[h, w] = make_u48(word)

# time_ref = np.zeros((P, P), dtype=np.int64)
# for w in range(P):
#     for h in range(P):
#         hex_t0 = int(float_to_hex(time_ref_hwl[h, w, 0]), base=16)
#         hex_t1 = int(float_to_hex(time_ref_hwl[h, w, 1]), base=16)
#         word[0] = hex_t0 & 0x0000ffff
#         word[1] = (hex_t0 >> 16) & 0x0000ffff
#         word[2] = hex_t1 & 0x0000ffff
#         time_ref[h, w] = make_u48(word)

# # Correct for per-PE clock skew and memcpy wavefront offset
# for py_i in range(P):
#     for px_i in range(P):
#         time_ref[py_i, px_i] -= px_i + py_i
# time_start = time_start - time_ref
# time_end   = time_end   - time_ref

# min_time_start = time_start.min()
# max_time_end   = time_end.max()

# print(f"\nRepeat count: {total_repeat_times}")
# print(f"P: {P}, M: {M}, K: {K}, N: {N}")
# print(f"Mean cycle count: {np.mean(time_end - time_start) / total_repeat_times:.1f}")
# print(f"Max  cycle count: {(max_time_end - min_time_start) / total_repeat_times:.1f}")

# freq_ghz  = 1.1
# time_cost = (max_time_end - min_time_start) / total_repeat_times / (freq_ghz * 1e6)
# print(f"Time: {time_cost:.4f} ms")

#!/usr/bin/env cs_python

# SUMMA Matrix Multiplication with Data Task Overlap
#
# Host program to:
# 1. Generate random A and B matrices
# 2. Distribute tiles to PEs
# 3. Run data-task-overlapped SUMMA kernel
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

with open(f"{args.name}/out.json", encoding='utf-8') as json_file:
    compile_data = json.load(json_file)

P = int(compile_data['params']['P'])
Mt = int(compile_data['params']['Mt'])
Kt = int(compile_data['params']['Kt'])
Nt = int(compile_data['params']['Nt'])

M = Mt * P
K = Kt * P
N = Nt * P

print(f"SUMMA with Data Task Overlap")
print(f"  Grid: {P} x {P} PEs")
print(f"  Tile sizes: Mt={Mt}, Kt={Kt}, Nt={Nt}")
print(f"  Full matrices: A({M}x{K}) @ B({K}x{N}) = C({M}x{N})")

memcpy_dtype = MemcpyDataType.MEMCPY_32BIT
memcpy_order = MemcpyOrder.ROW_MAJOR

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

print("Running data-task-overlapped SUMMA kernel...")
runner.launch("main", nonblock=False)

print("Copying C from device...")
C3_1d_u32 = np.zeros(h*w*Mt*Nt, np.uint32)
runner.memcpy_d2h(C3_1d_u32, sym_C, 0, 0, w, h, Mt*Nt,
    streaming=False, data_type=memcpy_dtype, order=MemcpyOrder.ROW_MAJOR, nonblock=False)

C3 = C3_1d_u32.reshape((h, w, Nt, Mt))
C2 = C3.transpose(0, 3, 1, 2)
C1 = C2.reshape(M, N)
C = C1.view(np.float32)

runner.stop()

print("Verifying results...")
C_expected = np.dot(A, B)

def is_error(a, b):
    return (np.abs((a - b) / (a + b)) > 1e-3) & (np.abs(a - b) > 0.05)

error_mask    = is_error(C_expected, C)
error_indices = np.argwhere(error_mask)
num_errors    = len(error_indices)

for idx in error_indices[:10]:
    i, j = idx
    print(f"  Mismatch at [{i},{j}]: expected {C_expected[i,j]:.6f}, got {C[i,j]:.6f}")

if num_errors == 0:
    print("SUCCESS")
else:
    print(f"FAILED: {num_errors} / {M*N} ({100 * num_errors/(M * N):.1f}%) elements mismatched")