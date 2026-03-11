#!/usr/bin/env cs_python

# Host program for MeshGEMM on WSE-3.
# Uses cerebras.sdk.runtime.sdkruntimepybind (standard appliance API).
# Replaces launch_wse3.py which used the deprecated cerebras.sdk.client API.

import argparse
import json
import struct
import numpy as np

from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime      # pylint: disable=no-name-in-module
from cerebras.sdk.runtime.sdkruntimepybind import MemcpyDataType  # pylint: disable=no-name-in-module
from cerebras.sdk.runtime.sdkruntimepybind import MemcpyOrder     # pylint: disable=no-name-in-module


def float_to_hex(f):
    return hex(struct.unpack('<I', struct.pack('<f', f))[0])


def make_u48(words):
    return int(words[0]) + (int(words[1]) << 16) + (int(words[2]) << 32)


def assignId(pc, P):
    """Compute the two-hop send/recv PE indices for the W pre-permutation."""
    send_id = 0
    recv_id = 0

    pc = pc + 1

    if pc % 2 == 0:
        send_id = pc - 2
        recv_id = pc + 2
    else:
        send_id = pc + 2
        recv_id = pc - 2
    if pc == 1:
        send_id = 3
        recv_id = 2

    if pc == 2:
        send_id = 1
        recv_id = min(recv_id, P)
    if P % 2 == 0:
        if pc == P - 1:
            send_id = P
            recv_id = P - 3
        if pc == P:
            send_id = P - 2
            recv_id = P - 1
    else:
        if pc == P - 1:
            send_id = max(send_id, 1)
            recv_id = P
        if pc == P:
            send_id = P - 1
            recv_id = P - 2
    return send_id - 1, recv_id - 1


parser = argparse.ArgumentParser(description="MeshGEMM host program (sdkruntimepybind)")
parser.add_argument("--name",      required=True,           help="Compiled output directory (e.g. 'out')")
parser.add_argument("--cmaddr",    default=None,            help="IP:port for CS system (omit for simulator)")
parser.add_argument("--P",         required=True, type=int, help="PE grid size P x P")
parser.add_argument("--M",         required=True, type=int, help="Input context length")
parser.add_argument("--K",         required=True, type=int, help="Word vector dimension")
parser.add_argument("--N",         required=True, type=int, help="Output dimension")
parser.add_argument("--warmup",    default=5,  type=int,   help="Warmup runs (default 5)")
parser.add_argument("--repeats",   default=50, type=int,   help="Timed runs (default 50)")
args = parser.parse_args()

P  = args.P
M  = args.M
K  = args.K
N  = args.N
Mt = M // P
Kt = K // P
Nt = N // P

np.random.seed(2025)
tensor_X = np.random.rand(M, K).astype(np.float16)
tensor_W = np.random.rand(K, N).astype(np.float16)

# Pre-permute W tiles to account for the two-hop shift pattern
ind = np.zeros((P, P), dtype=int)
for i in range(P):
    for j in range(P):
        if i == 0:
            ind[0, j] = j
        elif i == 1:
            _, ind[1, j] = assignId(ind[0, j], P)
        else:
            if (i - 1) % 2 == 0:
                _, ind[i, j] = assignId(ind[i - 2, j], P)
            else:
                ind[i, j], _ = assignId(ind[i - 2, j], P)

tensor_W_offset = np.zeros((K, N), dtype=np.float16)
for i in range(P):
    for j in range(P):
        t = ind[i, j]
        tensor_W_offset[i*Kt:(i+1)*Kt, j*Nt:(j+1)*Nt] = tensor_W[t*Kt:(t+1)*Kt, j*Nt:(j+1)*Nt]

# sdkruntimepybind MEMCPY_16BIT: 1 f16 element per uint32 word (low 16 bits used).
# count = number of f16 elements per PE (same as MEMCPY_32BIT convention).
# h2d: zero-extend each f16 to uint32 via view(uint16).astype(uint32).
# d2h: extract low 16 bits via astype(uint16), then view as float16.
io_dtype     = MemcpyDataType.MEMCPY_16BIT
memcpy_order = MemcpyOrder.ROW_MAJOR

runner = SdkRuntime(args.name, cmaddr=args.cmaddr)

sym_X             = runner.get_id("X")
sym_W             = runner.get_id("W")
sym_res           = runner.get_id("res")
sym_time_memcpy   = runner.get_id("time_memcpy")
sym_time_ref      = runner.get_id("time_ref")

runner.load()
runner.run()

res_1d_u32 = None

try:
    # Distribute X tiles: reshape into (P, P, Mt*Kt) in column-major tile order.
    # Zero-extend each f16 to uint32 (1 element per word) for MEMCPY_16BIT.
    X1    = tensor_X.reshape(P, Mt, P, Kt)
    X2    = X1.transpose(0, 2, 3, 1)
    X3    = X2.reshape(P, P, Mt * Kt)
    X_u32 = X3.ravel().view(np.uint16).astype(np.uint32)  # 1 uint32 per f16
    runner.memcpy_h2d(sym_X, X_u32, 0, 0, P, P, Mt * Kt,
                      streaming=False, data_type=io_dtype,
                      order=memcpy_order, nonblock=False)

    # Distribute W tiles (pre-permuted for two-hop alignment)
    W1    = tensor_W_offset.reshape(P, Kt, P, Nt)
    W2    = W1.transpose(0, 2, 1, 3)
    W3    = W2.reshape(P, P, Kt * Nt)
    W_u32 = W3.ravel().view(np.uint16).astype(np.uint32)  # 1 uint32 per f16
    runner.memcpy_h2d(sym_W, W_u32, 0, 0, P, P, Kt * Nt,
                      streaming=False, data_type=io_dtype,
                      order=memcpy_order, nonblock=False)

    # --- Pass 1: correctness check (0 warmup, 1 repeat) ---
    runner.launch('init_task', nonblock=False)
    total_warmup_times, total_repeat_times = 0, 1
    runner.launch('meshgemm_host',
                  np.int16(total_warmup_times), np.int16(total_repeat_times),
                  nonblock=False)

    # ── Correctness check d2h (small grids) ─────────────────────────
    # if P < 32: # only have verification checks for P < 32
    res_1d_u32 = np.zeros(P * P * Mt * Nt, dtype=np.uint32)
    runner.memcpy_d2h(res_1d_u32, sym_res, 0, 0, P, P, Mt * Nt,
                        streaming=False, data_type=io_dtype,
                        order=memcpy_order, nonblock=False)

    # --- Pass 2: performance timing ---
    runner.launch('init_task', nonblock=False)
    total_warmup_times, total_repeat_times = args.warmup, args.repeats
    runner.launch('meshgemm_host',
                  np.int16(total_warmup_times), np.int16(total_repeat_times),
                  nonblock=False)

    # Read cycle-count timestamps
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

# ── Correctness check ────────────────────────────────────────────────────────
res_fp16 = res_1d_u32.astype(np.uint16).view(np.float16)
res3 = res_fp16.reshape(P, P, Nt, Mt)
res2 = res3.transpose(0, 3, 1, 2)   # (P, Mt, P, Nt)
res  = res2.reshape(M, N)
C_ref = np.dot(tensor_X.astype(np.float32), tensor_W.astype(np.float32)).astype(np.float16)

# fp16 can have large relative errors; use rtol=0.5 matching Cerebras convention (see fft-3d/run.py)
try:
    np.testing.assert_allclose(res, C_ref, rtol=0.5, atol=0)
    print("Correctness check PASSED")
except AssertionError as e:
    print(f"Correctness check FAILED: {e}")

# Unpack 48-bit cycle counts from f32 bit patterns
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
