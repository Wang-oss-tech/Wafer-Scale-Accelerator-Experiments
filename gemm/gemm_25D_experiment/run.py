#!/usr/bin/env cs_python

# SUMMA Matrix Multiplication with 4-Color Overlapped Broadcasts — fp16 optimized variant
#
# Adapted from summa_manual_multicasting_pipelined_doubleColor-fp16.
# Key difference: B distributed row-major (transpose(0,2,1,3)) to match
# the @map-based do_gemm which accesses B[kk*Nt+j] (contiguous row kk).
#
# Two-pass execution:
#   Pass 1: summa_host(0, 1)  — correctness check
#   Pass 2: summa_host(warmup, repeats) — performance timing
#
# f16 memcpy convention (MEMCPY_16BIT):
#   h2d: view f16 as uint16, zero-extend to uint32 (1 element per word)
#   d2h: truncate uint32 to uint16, view as float16

import argparse
import json
import numpy as np

from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime
from cerebras.sdk.runtime.sdkruntimepybind import MemcpyDataType
from cerebras.sdk.runtime.sdkruntimepybind import MemcpyOrder


def f32_to_u32(x):
    return int(np.float32(x).view(np.uint32))


def make_u48(words):
    return int(words[0]) + (int(words[1]) << 16) + (int(words[2]) << 32)


parser = argparse.ArgumentParser(description="SUMMA fp16 4-color pipelined host program")
parser.add_argument("--name",    required=True, help="compiled output directory")
parser.add_argument("--cmaddr", default=None,   help="IP:port for CS system (omit for simulator)")
parser.add_argument("--P", default=None, type=int, help="PE grid size P x P")
parser.add_argument("--M", default=None, type=int, help="Full matrix M dimension")
parser.add_argument("--K", default=None, type=int, help="Full matrix K dimension")
parser.add_argument("--N", default=None, type=int, help="Full matrix N dimension")
parser.add_argument("--warmup",  default=5,  type=int, help="warmup runs (default 5)")
parser.add_argument("--repeats", default=50, type=int, help="timed runs (default 50)")
args = parser.parse_args()

if args.P is not None:
    P  = args.P
    M  = args.M
    K  = args.K
    N  = args.N
    Mt = M // P
    Kt = K // P
    Nt = N // P
else:
    with open(f"{args.name}/out.json", encoding='utf-8') as json_file:
        compile_data = json.load(json_file)
    P  = int(compile_data['params']['P'])
    Mt = int(compile_data['params']['Mt'])
    Kt = int(compile_data['params']['Kt'])
    Nt = int(compile_data['params']['Nt'])
    M = Mt * P
    K = Kt * P
    N = Nt * P

print(f"SUMMA with 4-Color Overlapped Broadcasts (fp16)")
print(f"  Grid: {P} x {P} PEs")
print(f"  Tile sizes: Mt={Mt}, Kt={Kt}, Nt={Nt}")
print(f"  Full matrices: A({M}x{K}) @ B({K}x{N}) = C({M}x{N})")

io_dtype     = MemcpyDataType.MEMCPY_16BIT
memcpy_order = MemcpyOrder.ROW_MAJOR

np.random.seed(seed=7)
A = np.random.rand(M, K).astype(np.float16)
B = np.random.rand(K, N).astype(np.float16)

# Cliff distribution — column-major local tiles
# h2d: view f16 as uint16, zero-extend to uint32
A1 = A.reshape(P, Mt, P, Kt)
A2 = A1.transpose(0, 2, 3, 1)       # (P, P, Kt, Mt) col-major local
A3 = A2.reshape(P, P, Mt * Kt)
A_u32 = A3.ravel().view(np.uint16).astype(np.uint32)

B1 = B.reshape(P, Kt, P, Nt)
B2 = B1.transpose(0, 2, 1, 3)       # (P, P, Kt, Nt) row-major local — matches do_gemm B[kk*Nt+j]
B3 = B2.reshape(P, P, Kt * Nt)
B_u32 = B3.ravel().view(np.uint16).astype(np.uint32)

runner = SdkRuntime(args.name, cmaddr=args.cmaddr)

sym_A           = runner.get_id("A")
sym_B           = runner.get_id("B")
sym_C           = runner.get_id("C")
sym_time_memcpy = runner.get_id("time_memcpy")
sym_time_ref    = runner.get_id("time_ref")

runner.load()
runner.run()

C3_1d_u32 = None

try:
    # Send input tiles (once; tiles stay on device across both passes)
    runner.memcpy_h2d(sym_A, A_u32, 0, 0, P, P, Mt * Kt,
                      streaming=False, data_type=io_dtype,
                      order=memcpy_order, nonblock=False)
    runner.memcpy_h2d(sym_B, B_u32, 0, 0, P, P, Kt * Nt,
                      streaming=False, data_type=io_dtype,
                      order=memcpy_order, nonblock=False)

    # --- Pass 1: correctness check (0 warmup, 1 repeat) ---
    total_warmup_times, total_repeat_times = 0, 1
    runner.launch("summa_host", np.int16(total_warmup_times), np.int16(total_repeat_times), nonblock=False)

    C3_1d_u32 = np.zeros(P * P * Mt * Nt, dtype=np.uint32)
    runner.memcpy_d2h(C3_1d_u32, sym_C, 0, 0, P, P, Mt * Nt,
                      streaming=False, data_type=io_dtype,
                      order=memcpy_order, nonblock=False)

    # --- Pass 2: performance timing ---
    if args.repeats > 0:
        total_warmup_times, total_repeat_times = args.warmup, args.repeats
        runner.launch("summa_host", np.int16(total_warmup_times), np.int16(total_repeat_times), nonblock=False)

        time_memcpy_1d_f32 = np.zeros(P * P * 3, dtype=np.float32)
        runner.memcpy_d2h(time_memcpy_1d_f32, sym_time_memcpy, 0, 0, P, P, 3,
                          streaming=False, data_type=MemcpyDataType.MEMCPY_32BIT,
                          order=MemcpyOrder.ROW_MAJOR, nonblock=False)
        time_memcpy_hwl = np.reshape(time_memcpy_1d_f32, (P, P, 3), order="C")

        time_ref_1d_f32 = np.zeros(P * P * 2, dtype=np.float32)
        runner.memcpy_d2h(time_ref_1d_f32, sym_time_ref, 0, 0, P, P, 2,
                          streaming=False, data_type=MemcpyDataType.MEMCPY_32BIT,
                          order=MemcpyOrder.ROW_MAJOR, nonblock=False)
        time_ref_hwl = np.reshape(time_ref_1d_f32, (P, P, 2), order="C")

finally:
    runner.stop()

# ── Correctness check ─────────────────────────────────────────────────────────
# d2h: truncate uint32 to uint16, view as float16
C_fp16 = C3_1d_u32.astype(np.uint16).view(np.float16)
C3 = C_fp16.reshape(P, P, Nt, Mt)
C2 = C3.transpose(0, 3, 1, 2)   # (P, Mt, P, Nt)
C  = C2.reshape(M, N)

# Reference: compute in f32, cast result to f16
C_expected = np.dot(A.astype(np.float32), B.astype(np.float32)).astype(np.float16)

np.testing.assert_allclose(C_expected, C, rtol=0.5, atol=0)
print("Correctness check PASSED")

if args.repeats <= 0:
    exit(0)

# ── Unpack 48-bit cycle counts ────────────────────────────────────────────────
time_start = np.zeros((P, P), dtype=np.int64)
time_end   = np.zeros((P, P), dtype=np.int64)
word = np.zeros(3, dtype=np.uint16)
for w in range(P):
    for h in range(P):
        hex_t0 = f32_to_u32(time_memcpy_hwl[h, w, 0])
        hex_t1 = f32_to_u32(time_memcpy_hwl[h, w, 1])
        hex_t2 = f32_to_u32(time_memcpy_hwl[h, w, 2])
        word[0] = hex_t0 & 0x0000ffff
        word[1] = (hex_t0 >> 16) & 0x0000ffff
        word[2] = hex_t1 & 0x0000ffff
        time_start[h, w] = make_u48(word)
        word[0] = (hex_t1 >> 16) & 0x0000ffff
        word[1] = hex_t2 & 0x0000ffff
        word[2] = (hex_t2 >> 16) & 0x0000ffff
        time_end[h, w] = make_u48(word)

time_ref = np.zeros((P, P), dtype=np.int64)
for w in range(P):
    for h in range(P):
        hex_t0 = f32_to_u32(time_ref_hwl[h, w, 0])
        hex_t1 = f32_to_u32(time_ref_hwl[h, w, 1])
        word[0] = hex_t0 & 0x0000ffff
        word[1] = (hex_t0 >> 16) & 0x0000ffff
        word[2] = hex_t1 & 0x0000ffff
        time_ref[h, w] = make_u48(word)

# Correct for per-PE clock skew and memcpy wavefront offset
for py_i in range(P):
    for px_i in range(P):
        time_ref[py_i, px_i] -= px_i + py_i
time_start = time_start - time_ref
time_end   = time_end   - time_ref

min_time_start = time_start.min()
max_time_end   = time_end.max()

print(f"\nRepeat count: {total_repeat_times}")
print(f"P: {P}, M: {M}, K: {K}, N: {N}")
print(f"Mean cycle count: {np.mean(time_end - time_start) / total_repeat_times:.1f}")
print(f"Max  cycle count: {(max_time_end - min_time_start) / total_repeat_times:.1f}")

freq_ghz  = 1.1
time_cost = (max_time_end - min_time_start) / total_repeat_times / (freq_ghz * 1e6)
print(f"Time: {time_cost:.4f} ms")
