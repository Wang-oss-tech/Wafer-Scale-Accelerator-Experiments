#!/usr/bin/env cs_python

# 2.5D GEMM host program for WSE-3.
#
# Each PE pre-loads c=2 A+B tile pairs for the local SUMMA steps.
# Remaining P-c steps use @fmovh broadcast (no host involvement).
#
# f16 memcpy convention (MEMCPY_16BIT):
#   h2d: view f16 as uint16, zero-extend to uint32 (1 element per word)
#   d2h: truncate uint32 to uint16, view as float16

import argparse
import struct
import numpy as np

from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime      # pylint: disable=no-name-in-module
from cerebras.sdk.runtime.sdkruntimepybind import MemcpyDataType  # pylint: disable=no-name-in-module
from cerebras.sdk.runtime.sdkruntimepybind import MemcpyOrder     # pylint: disable=no-name-in-module


LOCAL_STEPS = 2


def f32_to_u32(x):
    return int(np.float32(x).view(np.uint32))


def make_u48(words):
    return int(words[0]) + (int(words[1]) << 16) + (int(words[2]) << 32)


parser = argparse.ArgumentParser(description="2.5D GEMM host program (WSE-3)")
parser.add_argument("--name",    required=True,           help="Compiled output directory (e.g. 'out')")
parser.add_argument("--cmaddr", default=None,             help="IP:port for CS system (omit for simulator)")
parser.add_argument("--P",      required=True, type=int,  help="PE grid size P x P")
parser.add_argument("--M",      required=True, type=int,  help="Full M dimension")
parser.add_argument("--K",      required=True, type=int,  help="Full K dimension")
parser.add_argument("--N",      required=True, type=int,  help="Full N dimension")
parser.add_argument("--warmup",  default=5,  type=int,    help="Warmup runs (default 5)")
parser.add_argument("--repeats", default=50, type=int,    help="Timed runs (default 50)")
args = parser.parse_args()

P  = args.P
M  = args.M
K  = args.K
N  = args.N
Mt = M // P
Kt = K // P
Nt = N // P

assert M % P == 0, f"M={M} must be divisible by P={P}"
assert K % P == 0, f"K={K} must be divisible by P={P}"
assert N % P == 0, f"N={N} must be divisible by P={P}"

io_dtype     = MemcpyDataType.MEMCPY_16BIT
memcpy_order = MemcpyOrder.ROW_MAJOR

np.random.seed(seed=7)
A = np.random.rand(M, K).astype(np.float16)
B = np.random.rand(K, N).astype(np.float16)

# ── Cliff distribution — column-major local tiles ────────────────────────────
# A: each PE(px, py) owns A[py*Mt:(py+1)*Mt, px*Kt:(px+1)*Kt] stored col-major
A1  = A.reshape(P, Mt, P, Kt)
A2  = A1.transpose(0, 2, 3, 1)        # (P, P, Kt, Mt) col-major local
A3  = A2.reshape(P, P, Mt * Kt)
A_u32 = A3.ravel().view(np.uint16).astype(np.uint32)

# B: each PE(px, py) owns B[py*Kt:(py+1)*Kt, px*Nt:(px+1)*Nt] stored row-major
# (row-major matches the @map-based do_gemm that accesses B[kk*Nt+j])
B1  = B.reshape(P, Kt, P, Nt)
B2  = B1.transpose(0, 2, 1, 3)        # (P, P, Kt, Nt) row-major local
B3  = B2.reshape(P, P, Kt * Nt)
B_u32 = B3.ravel().view(np.uint16).astype(np.uint32)

# ── Pre-load A_local for c=2 local steps ─────────────────────────────────────
# A_local of PE(px, py) =
#   [0 : Mt*Kt]         : A tile for SUMMA step 0 = A[py*Mt:(py+1)*Mt, 0:Kt]   (col 0)
#   [Mt*Kt : 2*Mt*Kt]   : A tile for SUMMA step 1 = A[py*Mt:(py+1)*Mt, Kt:2*Kt] (col 1)
#
# Both tiles stored col-major to match do_gemm_ptrs.
# Construct (P, P, 2*Mt*Kt) array: axis-0 = py, axis-1 = px, axis-2 = elements.
A_local_np = np.zeros((P, P, LOCAL_STEPS * Mt * Kt), dtype=np.float16)
for s in range(LOCAL_STEPS):
    # All PEs in row py use the A tile from column s at SUMMA step s.
    # Tile = A[py*Mt:(py+1)*Mt, s*Kt:(s+1)*Kt], stored col-major (Kt, Mt).
    for py_i in range(P):
        tile = A[py_i * Mt:(py_i + 1) * Mt, s * Kt:(s + 1) * Kt]  # (Mt, Kt)
        tile_colmajor = tile.T.ravel()  # (Kt*Mt,) col-major
        for px_i in range(P):
            A_local_np[py_i, px_i, s * Mt * Kt:(s + 1) * Mt * Kt] = tile_colmajor
A_local_u32 = A_local_np.ravel().view(np.uint16).astype(np.uint32)

# ── Pre-load B_local for c=2 local steps ─────────────────────────────────────
# B_local of PE(px, py) =
#   [0 : Kt*Nt]         : B tile for SUMMA step 0 = B[0:Kt, px*Nt:(px+1)*Nt]   (row 0)
#   [Kt*Nt : 2*Kt*Nt]   : B tile for SUMMA step 1 = B[Kt:2*Kt, px*Nt:(px+1)*Nt] (row 1)
#
# Both tiles stored row-major (Kt, Nt) to match do_gemm_ptrs.
B_local_np = np.zeros((P, P, LOCAL_STEPS * Kt * Nt), dtype=np.float16)
for s in range(LOCAL_STEPS):
    for px_i in range(P):
        tile = B[s * Kt:(s + 1) * Kt, px_i * Nt:(px_i + 1) * Nt]  # (Kt, Nt) row-major
        tile_rowmajor = tile.ravel()
        for py_i in range(P):
            B_local_np[py_i, px_i, s * Kt * Nt:(s + 1) * Kt * Nt] = tile_rowmajor
B_local_u32 = B_local_np.ravel().view(np.uint16).astype(np.uint32)

# ── Runner setup ─────────────────────────────────────────────────────────────
runner = SdkRuntime(args.name, cmaddr=args.cmaddr)

sym_A           = runner.get_id("A")
sym_B           = runner.get_id("B")
sym_C           = runner.get_id("C")
sym_A_local     = runner.get_id("A_local")
sym_B_local     = runner.get_id("B_local")
sym_time_memcpy = runner.get_id("time_memcpy")
sym_time_ref    = runner.get_id("time_ref")

runner.load()
runner.run()

C3_1d_u32 = None

try:
    # Send home tiles (A, B) — used by broadcast-step roots
    runner.memcpy_h2d(sym_A, A_u32, 0, 0, P, P, Mt * Kt,
                      streaming=False, data_type=io_dtype,
                      order=memcpy_order, nonblock=False)
    runner.memcpy_h2d(sym_B, B_u32, 0, 0, P, P, Kt * Nt,
                      streaming=False, data_type=io_dtype,
                      order=memcpy_order, nonblock=False)
    # Send pre-loaded local tiles
    runner.memcpy_h2d(sym_A_local, A_local_u32, 0, 0, P, P, LOCAL_STEPS * Mt * Kt,
                      streaming=False, data_type=io_dtype,
                      order=memcpy_order, nonblock=False)
    runner.memcpy_h2d(sym_B_local, B_local_u32, 0, 0, P, P, LOCAL_STEPS * Kt * Nt,
                      streaming=False, data_type=io_dtype,
                      order=memcpy_order, nonblock=False)

    # --- Pass 1: correctness check (0 warmup, 1 repeat) ---
    total_warmup_times, total_repeat_times = 0, 1
    runner.launch("summa_host", np.int16(total_warmup_times), np.int16(total_repeat_times),
                  nonblock=False)

    C3_1d_u32 = np.zeros(P * P * Mt * Nt, dtype=np.uint32)
    runner.memcpy_d2h(C3_1d_u32, sym_C, 0, 0, P, P, Mt * Nt,
                      streaming=False, data_type=io_dtype,
                      order=memcpy_order, nonblock=False)

    # --- Pass 2: performance timing ---
    if args.repeats > 0:
        total_warmup_times, total_repeat_times = args.warmup, args.repeats
        runner.launch("summa_host", np.int16(total_warmup_times), np.int16(total_repeat_times),
                      nonblock=False)

        time_memcpy_1d_f32 = np.zeros(P * P * 3, dtype=np.float32)
        runner.memcpy_d2h(time_memcpy_1d_f32, sym_time_memcpy, 0, 0, P, P, 3,
                          streaming=False, data_type=MemcpyDataType.MEMCPY_32BIT,
                          order=MemcpyOrder.ROW_MAJOR, nonblock=False)
        time_memcpy_hwl = np.reshape(time_memcpy_1d_f32, (P, P, 3), order='C')

        time_ref_1d_f32 = np.zeros(P * P * 2, dtype=np.float32)
        runner.memcpy_d2h(time_ref_1d_f32, sym_time_ref, 0, 0, P, P, 2,
                          streaming=False, data_type=MemcpyDataType.MEMCPY_32BIT,
                          order=MemcpyOrder.ROW_MAJOR, nonblock=False)
        time_ref_hwl = np.reshape(time_ref_1d_f32, (P, P, 2), order='C')

finally:
    runner.stop()

# ── Correctness check ─────────────────────────────────────────────────────────
C_fp16 = C3_1d_u32.astype(np.uint16).view(np.float16)
C3  = C_fp16.reshape(P, P, Nt, Mt)
C2  = C3.transpose(0, 3, 1, 2)    # (P, Mt, P, Nt)
C   = C2.reshape(M, N)

C_ref = np.dot(A.astype(np.float32), B.astype(np.float32)).astype(np.float16)

try:
    np.testing.assert_allclose(C, C_ref, rtol=0.5, atol=0)
    print("Correctness check PASSED")
except AssertionError as e:
    print(f"Correctness check FAILED: {e}")

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
