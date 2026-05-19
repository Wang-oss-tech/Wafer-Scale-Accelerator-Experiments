#!/usr/bin/env cs_python

# WSE-3 Prefill host program.
# Adapted from WaferLLM/Prefill/WSE-2/launch_sim.py.
#
# Weight pre-shuffling (assignId/ind) is required for correctness:
# the two-hop communication pattern expects weights to be pre-positioned
# according to the checkerboard shift index before launch.

import argparse
import struct
import numpy as np

from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime
from cerebras.sdk.runtime.sdkruntimepybind import MemcpyDataType, MemcpyOrder


def float_to_hex(f):
    return hex(struct.unpack("<I", struct.pack("<f", f))[0])


def make_u48(words):
    return int(words[0]) + (int(words[1]) << 16) + (int(words[2]) << 32)


def to_u32(arr_f16):
    return arr_f16.ravel().view(np.uint16).astype(np.uint32)


def assignId(pc, P):
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


parser = argparse.ArgumentParser(description="WSE-3 Prefill host program")
parser.add_argument("--name",        required=True,           help="Compiled output directory (e.g. 'out')")
parser.add_argument("--cmaddr",      default=None,            help="IP:port for CS system (omit for simulator)")
parser.add_argument("--P",           required=True, type=int, help="PE grid size P x P")
parser.add_argument("--dim",         required=True, type=int, help="Model hidden dimension")
parser.add_argument("--seq_len",     required=True, type=int, help="Sequence length")
parser.add_argument("--ffn_dim",     required=True, type=int, help="FFN intermediate dimension")
parser.add_argument("--head_dim",    required=True, type=int, help="Attention head dimension")
parser.add_argument("--n_heads",     default=1,     type=int, help="Number of attention heads")
parser.add_argument("--n_kv_heads",  default=1,     type=int, help="Number of KV heads")
parser.add_argument("--warmup",      default=1,     type=int, help="Warmup runs (default 1)")
parser.add_argument("--repeats",     default=3,     type=int, help="Timed runs (default 3)")
parser.add_argument("--input_tokens", default=0,    type=int,
                    help="Actual prompt length (real tokens). 0 = same as seq_len (no padding-invariance test). "
                         "If set < seq_len, runs the kernel TWICE with same first input_tokens rows of X but "
                         "different padding in the last (seq_len - input_tokens) rows, and asserts the first "
                         "input_tokens rows of Z agree (within fp16 tolerance). Verifies the causal mask "
                         "isolates real outputs from padding inputs.")
args = parser.parse_args()

P          = args.P
dim        = args.dim
seq_len    = args.seq_len
ffn_dim    = args.ffn_dim

dim_p_pe     = dim // P
seq_len_p_pe = seq_len // P
ffn_dim_p_pe = ffn_dim // P

_dim_p_pe = dim_p_pe
if dim_p_pe % 2 == 1:
    _dim_p_pe = dim_p_pe - 1

io_dtype     = MemcpyDataType.MEMCPY_16BIT
memcpy_order = MemcpyOrder.ROW_MAJOR

# Scale random inputs to small magnitudes so the f16 matmul cascade
# (~dim multiply-adds compounded across rmsnorm/Q-K-V/attn/FFN) doesn't
# overflow f16 (max ~65504). Without scaling, large dim produces inf/NaN.
# See investigation: with U(0,1) inputs at dim=64+ we hit overflow.
INPUT_SCALE = 0.01

np.random.seed(7)

# Build the "real" portion of X (first input_tokens rows). If input_tokens
# is 0 or matches seq_len, this is just the full X.
input_tokens = args.input_tokens if args.input_tokens > 0 else seq_len
real_x = (np.random.rand(input_tokens, dim).astype(np.float16) * INPUT_SCALE).astype(np.float16)

# Padding-A: random pad with seed 11
np.random.seed(11)
pad_a = (np.random.rand(seq_len - input_tokens, dim).astype(np.float16) * INPUT_SCALE).astype(np.float16) if input_tokens < seq_len else None

# Padding-B: random pad with seed 13 (used only for invariance test second run)
np.random.seed(13)
pad_b = (np.random.rand(seq_len - input_tokens, dim).astype(np.float16) * INPUT_SCALE).astype(np.float16) if input_tokens < seq_len else None

tensor_X = real_x if pad_a is None else np.concatenate([real_x, pad_a], axis=0)

W = np.random.rand(1, dim).astype(np.float16)
tensor_W = np.tile(W.reshape(P, dim_p_pe), reps=(1, P))

WEIGHT_SCALE = 0.01
tensor_q_weight = (np.random.rand(dim, dim).astype(np.float16) * WEIGHT_SCALE).astype(np.float16)
tensor_k_weight = (np.random.rand(dim, dim).astype(np.float16) * WEIGHT_SCALE).astype(np.float16)
tensor_v_weight = (np.random.rand(dim, dim).astype(np.float16) * WEIGHT_SCALE).astype(np.float16)

freqs_sin = np.random.rand(1, P * _dim_p_pe // 2).astype(np.float16)
tensor_freqs_sin = np.tile(freqs_sin.reshape(P, _dim_p_pe // 2), reps=(1, P))
freqs_cos = np.random.rand(1, P * _dim_p_pe // 2).astype(np.float16)
tensor_freqs_cos = np.tile(freqs_cos.reshape(P, _dim_p_pe // 2), reps=(1, P))

tensor_o_weight    = (np.random.rand(dim, dim).astype(np.float16)     * WEIGHT_SCALE).astype(np.float16)
tensor_up_weight   = (np.random.rand(dim, ffn_dim).astype(np.float16) * WEIGHT_SCALE).astype(np.float16)
tensor_gate_weight = (np.random.rand(dim, ffn_dim).astype(np.float16) * WEIGHT_SCALE).astype(np.float16)
tensor_down_weight = (np.random.rand(ffn_dim, dim).astype(np.float16) * WEIGHT_SCALE).astype(np.float16)

# ── Weight pre-shuffling ─────────────────────────────────────────────────────
# Build shift index: ind[i,j] = source row in weight matrix for PE(j,i)
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

tensor_q_weight_shifted    = np.zeros((dim, dim),     dtype=np.float16)
tensor_k_weight_shifted    = np.zeros((dim, dim),     dtype=np.float16)
tensor_v_weight_shifted    = np.zeros((dim, dim),     dtype=np.float16)
tensor_o_weight_shifted    = np.zeros((dim, dim),     dtype=np.float16)
tensor_up_weight_shifted   = np.zeros((dim, ffn_dim), dtype=np.float16)
tensor_gate_weight_shifted = np.zeros((dim, ffn_dim), dtype=np.float16)
tensor_down_weight_shifted = np.zeros((ffn_dim, dim), dtype=np.float16)

for i in range(P):
    for j in range(P):
        t = ind[i, j]
        tensor_q_weight_shifted   [i*dim_p_pe:(i+1)*dim_p_pe, j*dim_p_pe:(j+1)*dim_p_pe]     = tensor_q_weight   [t*dim_p_pe:(t+1)*dim_p_pe, j*dim_p_pe:(j+1)*dim_p_pe]
        tensor_k_weight_shifted   [i*dim_p_pe:(i+1)*dim_p_pe, j*dim_p_pe:(j+1)*dim_p_pe]     = tensor_k_weight   [t*dim_p_pe:(t+1)*dim_p_pe, j*dim_p_pe:(j+1)*dim_p_pe]
        tensor_v_weight_shifted   [i*dim_p_pe:(i+1)*dim_p_pe, j*dim_p_pe:(j+1)*dim_p_pe]     = tensor_v_weight   [t*dim_p_pe:(t+1)*dim_p_pe, j*dim_p_pe:(j+1)*dim_p_pe]
        tensor_o_weight_shifted   [i*dim_p_pe:(i+1)*dim_p_pe, j*dim_p_pe:(j+1)*dim_p_pe]     = tensor_o_weight   [t*dim_p_pe:(t+1)*dim_p_pe, j*dim_p_pe:(j+1)*dim_p_pe]
        tensor_up_weight_shifted  [i*dim_p_pe:(i+1)*dim_p_pe, j*ffn_dim_p_pe:(j+1)*ffn_dim_p_pe] = tensor_up_weight  [t*dim_p_pe:(t+1)*dim_p_pe, j*ffn_dim_p_pe:(j+1)*ffn_dim_p_pe]
        tensor_gate_weight_shifted[i*dim_p_pe:(i+1)*dim_p_pe, j*ffn_dim_p_pe:(j+1)*ffn_dim_p_pe] = tensor_gate_weight[t*dim_p_pe:(t+1)*dim_p_pe, j*ffn_dim_p_pe:(j+1)*ffn_dim_p_pe]
        tensor_down_weight_shifted[i*ffn_dim_p_pe:(i+1)*ffn_dim_p_pe, j*dim_p_pe:(j+1)*dim_p_pe] = tensor_down_weight[t*ffn_dim_p_pe:(t+1)*ffn_dim_p_pe, j*dim_p_pe:(j+1)*dim_p_pe]

# ── Runner setup ─────────────────────────────────────────────────────────────
runner = SdkRuntime(args.name, cmaddr=args.cmaddr)

sym_X          = runner.get_id("X")
sym_Z          = runner.get_id("Z")
sym_W          = runner.get_id("W")
sym_Q_weight   = runner.get_id("Q_weight")
sym_K_weight   = runner.get_id("K_weight")
sym_V_weight   = runner.get_id("V_weight")
sym_freqs_sin  = runner.get_id("freqs_sin")
sym_freqs_cos  = runner.get_id("freqs_cos")
sym_O_weight   = runner.get_id("O_weight")
sym_UP_weight  = runner.get_id("UP_weight")
sym_GATE_weight = runner.get_id("GATE_weight")
sym_DOWN_weight = runner.get_id("DOWN_weight")
sym_time_memcpy = runner.get_id("time_memcpy")
sym_time_ref    = runner.get_id("time_ref")

runner.load()
runner.run()

# ── H2D transfers ────────────────────────────────────────────────────────────
# (X H2D done below via h2d_X helper, after helpers are defined.)

runner.memcpy_h2d(sym_W, to_u32(tensor_W), 0, 0, P, P, dim_p_pe,
                  streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False)

def send_weight_dim_dim(sym, w_shifted):
    r = w_shifted.reshape(P, dim_p_pe, P, dim_p_pe).transpose(0, 2, 1, 3).reshape(P, P, dim_p_pe * dim_p_pe)
    runner.memcpy_h2d(sym, to_u32(r), 0, 0, P, P, dim_p_pe * dim_p_pe,
                      streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False)

def send_weight_dim_ffn(sym, w_shifted):
    r = w_shifted.reshape(P, dim_p_pe, P, ffn_dim_p_pe).transpose(0, 2, 1, 3).reshape(P, P, dim_p_pe * ffn_dim_p_pe)
    runner.memcpy_h2d(sym, to_u32(r), 0, 0, P, P, dim_p_pe * ffn_dim_p_pe,
                      streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False)

def send_weight_ffn_dim(sym, w_shifted):
    r = w_shifted.reshape(P, ffn_dim_p_pe, P, dim_p_pe).transpose(0, 2, 1, 3).reshape(P, P, ffn_dim_p_pe * dim_p_pe)
    runner.memcpy_h2d(sym, to_u32(r), 0, 0, P, P, ffn_dim_p_pe * dim_p_pe,
                      streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False)

send_weight_dim_dim(sym_Q_weight, tensor_q_weight_shifted)
send_weight_dim_dim(sym_K_weight, tensor_k_weight_shifted)
send_weight_dim_dim(sym_V_weight, tensor_v_weight_shifted)
send_weight_dim_dim(sym_O_weight, tensor_o_weight_shifted)
send_weight_dim_ffn(sym_UP_weight,   tensor_up_weight_shifted)
send_weight_dim_ffn(sym_GATE_weight, tensor_gate_weight_shifted)
send_weight_ffn_dim(sym_DOWN_weight, tensor_down_weight_shifted)

runner.memcpy_h2d(sym_freqs_sin, to_u32(tensor_freqs_sin), 0, 0, P, P, _dim_p_pe // 2,
                  streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False)
runner.memcpy_h2d(sym_freqs_cos, to_u32(tensor_freqs_cos), 0, 0, P, P, _dim_p_pe // 2,
                  streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False)

# ── Helpers for the prefill iteration ────────────────────────────────────────

def h2d_X(x_tensor):
    """Send (seq_len, dim) f16 input to PE-distributed X tiles."""
    Xc1 = x_tensor.reshape(P, seq_len_p_pe, P, dim_p_pe)
    Xc2 = Xc1.transpose(0, 2, 3, 1)
    Xc3 = Xc2.reshape(P, P, seq_len_p_pe * dim_p_pe)
    runner.memcpy_h2d(sym_X, to_u32(Xc3), 0, 0, P, P, seq_len_p_pe * dim_p_pe,
                      streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False)

def d2h_Z():
    """Read (seq_len, dim) final-block output Z from device."""
    Z_u32 = np.zeros(P * P * seq_len_p_pe * dim_p_pe, dtype=np.uint32)
    runner.memcpy_d2h(Z_u32, sym_Z, 0, 0, P, P, seq_len_p_pe * dim_p_pe,
                      streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False)
    # Recover f16 from low 16 bits of each u32 slot.
    Z_f16_flat = Z_u32.astype(np.uint16, copy=False).view(np.float16)
    # Inverse of H2D layout: (P, P, seq_len_p_pe * dim_p_pe) → (seq_len, dim).
    Zc3 = Z_f16_flat.reshape(P, P, dim_p_pe, seq_len_p_pe)
    Zc2 = Zc3.transpose(0, 3, 1, 2)
    return Zc2.reshape(seq_len, dim)

# ── Launch (initial run with padding-A) ──────────────────────────────────────
h2d_X(tensor_X)
runner.launch('init_task', nonblock=False)
runner.launch('prefill_host', np.int16(args.warmup), np.int16(args.repeats), nonblock=False)
Z_a = d2h_Z()

# ── Padding-invariance test (second run with padding-B) ──────────────────────
# If input_tokens < seq_len, run again with the SAME first input_tokens rows of
# X but DIFFERENT random padding in the last (seq_len - input_tokens) rows.
# With causal mask working, real outputs at rows [0, input_tokens) must agree
# bit-for-bit (or within fp16 noise) — they only depend on inputs at [0, k]
# for query k, never on anything past input_tokens-1.
if input_tokens < seq_len:
    tensor_X_b = np.concatenate([real_x, pad_b], axis=0)
    h2d_X(tensor_X_b)
    runner.launch('prefill_host', np.int16(args.warmup), np.int16(args.repeats), nonblock=False)
    Z_b = d2h_Z()

    real_out_a = Z_a[:input_tokens].astype(np.float32)
    real_out_b = Z_b[:input_tokens].astype(np.float32)
    print("\n=== Padding-invariance test ===")
    print(f"Compared first {input_tokens} rows of Z under two different padding fills.")
    # Sanity check 1: outputs must be finite (no NaN/Inf), else any comparison
    # is meaningless. NaN+NaN+rtol passes silently in older numpy.
    assert not np.any(np.isnan(real_out_a)), "Z_a contains NaN — kernel produced invalid output (numerical overflow likely). Cannot evaluate padding invariance."
    assert not np.any(np.isnan(real_out_b)), "Z_b contains NaN — kernel produced invalid output."
    assert not np.any(np.isinf(real_out_a)), "Z_a contains Inf — kernel produced invalid output."
    assert not np.any(np.isinf(real_out_b)), "Z_b contains Inf — kernel produced invalid output."
    abs_diff = np.abs(real_out_a - real_out_b)
    rel_diff = abs_diff / (np.abs(real_out_a) + 1e-6)
    print(f"Z_a sample [0,:6]:    {real_out_a[0,:6]}")
    print(f"Z_b sample [0,:6]:    {real_out_b[0,:6]}")
    print(f"max abs diff (real): {abs_diff.max():.6f}, mean abs diff: {abs_diff.mean():.6f}")
    print(f"max rel diff (real): {rel_diff.max():.6f}, mean rel diff: {rel_diff.mean():.6f}")
    # Sanity: padding-region outputs SHOULD differ between runs (since padding
    # tokens see real KV via causal attention, and their input changed). If
    # they match, the second run isn't actually running prefill.
    pad_diff = np.abs(Z_a[input_tokens:].astype(np.float32) - Z_b[input_tokens:].astype(np.float32))
    print(f"max abs diff (padding region): {pad_diff.max():.6f} (should be > 0; sanity that 2nd run executed)")
    # Causal mask should make real outputs identical to within fp16 rounding.
    np.testing.assert_allclose(real_out_a, real_out_b, rtol=1e-2, atol=1e-3,
        err_msg=f"Padding-invariance FAILED: real outputs differ when only padding "
                f"changes. Causal mask is not isolating real tokens from padding.")
    print(f"PASSED: padding-invariance check (input_tokens={input_tokens}, seq_len={seq_len})")

# ── D2H timing ───────────────────────────────────────────────────────────────
time_memcpy_1d_f32 = np.zeros(P * P * 3, dtype=np.float32)
runner.memcpy_d2h(time_memcpy_1d_f32, sym_time_memcpy, 0, 0, P, P, 3,
                  streaming=False, order=MemcpyOrder.ROW_MAJOR,
                  data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)
time_memcpy_hwl = np.reshape(time_memcpy_1d_f32, (P, P, 3), order='C')

time_ref_1d_f32 = np.zeros(P * P * 2, dtype=np.float32)
runner.memcpy_d2h(time_ref_1d_f32, sym_time_ref, 0, 0, P, P, 2,
                  streaming=False, order=MemcpyOrder.ROW_MAJOR,
                  data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)
time_ref_hwl = np.reshape(time_ref_1d_f32, (P, P, 2), order='C')

runner.stop()

# ── Cycle count unpacking ─────────────────────────────────────────────────────
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

for py_i in range(P):
    for px_i in range(P):
        time_ref[py_i, px_i] -= px_i + py_i

time_start = time_start - time_ref
time_end   = time_end   - time_ref

min_time_start = time_start.min()
max_time_end   = time_end.max()

print(f"\nRepeat count: {args.repeats}")
print(f"P: {P}, dim: {dim}, seq_len: {seq_len}, ffn_dim: {ffn_dim}")
print(f"Mean cycle count: {np.mean(time_end - time_start) / args.repeats:.1f}")
print(f"Max  cycle count: {(max_time_end - min_time_start) / args.repeats:.1f}")

freq_ghz  = 1.1
time_cost = (max_time_end - min_time_start) / args.repeats / (freq_ghz * 1e6)
print(f"Time: {time_cost:.4f} ms")
