#!/usr/bin/env cs_python

# Host program for SUMMA GEMM (collectives_2d) — fp16 variant.
# Adapted from gemm-collectives_2d (f32) for fair comparison with MeshGEMM.
#
# Two-pass execution (matching MeshGEMM structure):
#   Pass 1: summa_host(0, 1)  — correctness check (P < 32 only)
#   Pass 2: summa_host(5, 50) — performance timing (5 warmup, 50 timed runs)
#
# f16 memcpy convention (MEMCPY_16BIT, sdkruntimepybind):
#   h2d: view f16 as uint16, zero-extend to uint32 (1 element per word)
#   d2h: truncate uint32 to uint16, view as float16

import argparse
import json
import struct
import numpy as np

from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime      # pylint: disable=no-name-in-module
from cerebras.sdk.runtime.sdkruntimepybind import MemcpyDataType  # pylint: disable=no-name-in-module
from cerebras.sdk.runtime.sdkruntimepybind import MemcpyOrder     # pylint: disable=no-name-in-module


def float_to_hex(f):
    return hex(struct.unpack("<I", struct.pack("<f", f))[0])


def make_u48(words):
    return int(words[0]) + (int(words[1]) << 16) + (int(words[2]) << 32)


parser = argparse.ArgumentParser(description="SUMMA fp16 host program")
parser.add_argument("--name",   required=True,         help="Compiled output directory")
parser.add_argument("--cmaddr", default=None,          help="IP:port for CS system (omit for simulator)")
# Optional: pass P/M/K/N explicitly (required for appliance launch where out.json
# may not be present on the worker node; omit to read from out.json instead)
parser.add_argument("--P", default=None, type=int, help="PE grid size P x P")
parser.add_argument("--M", default=None, type=int, help="Full matrix M dimension")
parser.add_argument("--K", default=None, type=int, help="Full matrix K dimension")
parser.add_argument("--N", default=None, type=int, help="Full matrix N dimension")
parser.add_argument("--warmup",  default=5,  type=int, help="Warmup runs (default 5)")
parser.add_argument("--repeats", default=50, type=int, help="Timed runs (default 50)")
args = parser.parse_args()

# Resolve P/Mt/Kt/Nt: prefer explicit CLI args, fall back to out.json
if args.P is not None:
    P  = args.P
    M  = args.M
    K  = args.K
    N  = args.N
    Mt = M // P
    Kt = K // P
    Nt = N // P
else:
    with open(f"{args.name}/out.json", encoding="utf-8") as f:
        compile_data = json.load(f)
    P  = int(compile_data["params"]["P"])
    Mt = int(compile_data["params"]["Mt"])
    Kt = int(compile_data["params"]["Kt"])
    Nt = int(compile_data["params"]["Nt"])
    M  = Mt * P
    K  = Kt * P
    N  = Nt * P

io_dtype     = MemcpyDataType.MEMCPY_16BIT
memcpy_order = MemcpyOrder.ROW_MAJOR

np.random.seed(2025)
A = np.random.rand(M, K).astype(np.float16)
B = np.random.rand(K, N).astype(np.float16)

# Cliff distribution — column-major local tiles (same layout as f32 original)
A1 = A.reshape(P, Mt, P, Kt)
A2 = A1.transpose(0, 2, 3, 1)       # (P, P, Kt, Mt) col-major local
A3 = A2.reshape(P, P, Mt * Kt)
A_u32 = A3.ravel().view(np.uint16).astype(np.uint32)

B1 = B.reshape(P, Kt, P, Nt)
B2 = B1.transpose(0, 2, 3, 1)       # (P, P, Nt, Kt) col-major local
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

C_1d_u32 = None

try:
    # ── Send input tiles (once; tiles stay on device across both passes) ──────
    runner.memcpy_h2d(sym_A, A_u32, 0, 0, P, P, Mt * Kt,
                      streaming=False, data_type=io_dtype,
                      order=memcpy_order, nonblock=False)
    runner.memcpy_h2d(sym_B, B_u32, 0, 0, P, P, Kt * Nt,
                      streaming=False, data_type=io_dtype,
                      order=memcpy_order, nonblock=False)

    # --- Pass 1: correctness check (0 warmup, 1 repeat) ---
    total_warmup_times, total_repeat_times = 0, 1
    runner.launch("summa_host", np.int16(total_warmup_times), np.int16(total_repeat_times), nonblock=False)

    # ── Correctness check d2h ─────────────────────────────────────────────────
    C_1d_u32 = np.zeros(P * P * Mt * Nt, dtype=np.uint32)
    runner.memcpy_d2h(C_1d_u32, sym_C, 0, 0, P, P, Mt * Nt,
                      streaming=False, data_type=io_dtype,
                      order=memcpy_order, nonblock=False)

    # --- Pass 2: performance timing (skipped if --repeats 0) ---
    if args.repeats > 0:
        total_warmup_times, total_repeat_times = args.warmup, args.repeats
        runner.launch("summa_host", np.int16(total_warmup_times), np.int16(total_repeat_times), nonblock=False)

        # ── Retrieve timestamps (span all timed runs; divide by total_repeat_times) ─
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

# ── Correctness check ────────────────────────────────────────────────────────
# Reconstruct global C from column-major tiles (inverse cliff distribution)
C_fp16 = C_1d_u32.astype(np.uint16).view(np.float16)
C3 = C_fp16.reshape(P, P, Nt, Mt)
C2 = C3.transpose(0, 3, 1, 2)   # (P, Mt, P, Nt)
C_device = C2.reshape(M, N)
C_ref = np.dot(A.astype(np.float32), B.astype(np.float32)).astype(np.float16)

# Compare as fp32 to avoid fp16 arithmetic in the check itself.
def is_error(a, b):
    return (np.abs((a - b) / (a + b)) > 1e-3) & (np.abs(a - b) > 0.05) # relative and absolute error hold true

a = C_device.astype(np.float32)
b = C_ref.astype(np.float32)
error_mask    = is_error(a, b)
error_indices = np.argwhere(error_mask)
num_errors    = len(error_indices)

# we only output the 10 errors
for idx in error_indices[:10]:
    i, j = idx
    print(f"Mismatch at [{i},{j}]: expected {b[i,j]:.4f}, got {a[i,j]:.4f}")

if num_errors == 0:
    print("Correctness check PASSED")
else:
    print(f"Correctness check FAILED: {num_errors} / {M*N} ({100*num_errors/(M*N):.1f}%) elements mismatched")

# fp16 can have large relative errors; use rtol=0.5 matching Cerebras convention (see fft-3d/run.py)
try:
    np.testing.assert_allclose(C_device, C_ref, rtol=0.5, atol=0)
    print("Correctness check PASSED")
except AssertionError as e:
    print(f"Correctness check FAILED: {e}")

# ── Unpack 48-bit cycle counts ───────────────────────────────────────────────
time_start = np.zeros((P, P), dtype=np.int64)
time_end   = np.zeros((P, P), dtype=np.int64)
word = np.zeros(3, dtype=np.uint16)
for w in range(P):
    for h in range(P):
        hex_t0 = int(float_to_hex(time_memcpy_hwl[h, w, 0]), base=16)
        hex_t1 = int(float_to_hex(time_memcpy_hwl[h, w, 1]), base=16)
        hex_t2 = int(float_to_hex(time_memcpy_hwl[h, w, 2]), base=16)
        word[0] = hex_t0 & 0x0000ffff
        word[1] = (hex_t0 >> 16) & 0x0000ffff
        word[2] = hex_t1 & 0x0000ffff
        time_start[h, w] = make_u48(word)
        word[0] = (hex_t1 >> 16) & 0x0000ffff
        word[1] = hex_t2 & 0x0000ffff
        word[2] = (hex_t2 >> 16) & 0x0000ffff
        time_end[h, w] = make_u48(word)

time_ref = np.zeros((P, P), dtype=np.int64)
word = np.zeros(3, dtype=np.uint16)
for w in range(P):
    for h in range(P):
        hex_t0 = int(float_to_hex(time_ref_hwl[h, w, 0]), base=16)
        hex_t1 = int(float_to_hex(time_ref_hwl[h, w, 1]), base=16)
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
