import argparse
import json
import struct
import numpy as np

from cerebras.sdk.client import SdkRuntime, sdk_utils
from cerebras.appliance.pb.sdk.sdk_common_pb2 import MemcpyDataType, MemcpyOrder


def float_to_hex(f):
    return hex(struct.unpack('<I', struct.pack('<f', f))[0])

def make_u48(words):
    return words[0] + (words[1] << 16) + (words[2] << 32)

out_path = "compile_out"


def parse_args():
    parser = argparse.ArgumentParser(description="fp16 GEMV hardware runner")
    parser.add_argument("--P",         required=True, type=int, help="PE grid size P x P")
    parser.add_argument("--M",         required=True, type=int, help="Matrix rows (output dim)")
    parser.add_argument("--N",         required=True, type=int, help="Matrix cols (input dim)")
    parser.add_argument("--simulator", action="store_true", help="Run on simulator")
    return parser.parse_args()


args = parse_args()
P = args.P
M = args.M
N = args.N

with open(f"{out_path}/artifact_{P}_{M}_{N}.json", "r", encoding="utf-8") as f:
    artifact_id = json.load(f)["artifact_id"]

memcpy_dtype_16 = MemcpyDataType.MEMCPY_16BIT
memcpy_dtype_32 = MemcpyDataType.MEMCPY_32BIT
memcpy_order    = MemcpyOrder.ROW_MAJOR

np.random.seed(seed=7)
A = np.random.rand(M, N).astype(np.float16)
X = np.random.rand(N).astype(np.float16)
B = np.random.rand(M).astype(np.float16)

y_expected = (A.astype(np.float32) @ X.astype(np.float32)) + B.astype(np.float32)

kernel_rows = P
kernel_cols = P
per_pe_rows = M // P
per_pe_cols = N // P

# Column-major local tile distribution
A1   = A.reshape(kernel_rows, per_pe_rows, kernel_cols, per_pe_cols)
A2   = A1.transpose(0, 2, 3, 1)  # (h, w, Nt, Mt)
data = A2.reshape(kernel_rows, kernel_cols, per_pe_rows * per_pe_cols).ravel()
A_u32 = sdk_utils.input_array_to_u32(data, 1, 1)

x_u32 = sdk_utils.input_array_to_u32(X, 1, 1)
b_u32 = sdk_utils.input_array_to_u32(B, 1, 1)

print("Start running...", flush=True)
with SdkRuntime(artifact_id, simulator=args.simulator, disable_version_check=True) as runner:

    symbol_A           = runner.get_id("A")
    symbol_x           = runner.get_id("x")
    symbol_b           = runner.get_id("b")
    symbol_y           = runner.get_id("y")
    symbol_time_memcpy = runner.get_id("time_memcpy")
    symbol_time_ref    = runner.get_id("time_ref")

    runner.memcpy_h2d(symbol_A, A_u32, 0, 0, kernel_cols, kernel_rows, per_pe_rows * per_pe_cols,
                      streaming=False, data_type=memcpy_dtype_16, nonblock=False, order=memcpy_order)
    runner.memcpy_h2d(symbol_x, x_u32, 0, 0, 1, 1, N,
                      streaming=False, data_type=memcpy_dtype_16, nonblock=False, order=memcpy_order)
    runner.memcpy_h2d(symbol_b, b_u32, 0, 0, 1, 1, M,
                      streaming=False, data_type=memcpy_dtype_16, nonblock=False, order=memcpy_order)

    runner.launch("main", nonblock=False)

    y = np.zeros(M, dtype=np.float32)
    runner.memcpy_d2h(y, symbol_y, kernel_cols - 1, kernel_rows - 1, 1, 1, M,
                      streaming=False, data_type=memcpy_dtype_32, nonblock=False, order=memcpy_order)

    time_memcpy_1d = np.zeros(P * P * 3, dtype=np.float32)
    runner.memcpy_d2h(time_memcpy_1d, symbol_time_memcpy, 0, 0, P, P, 3,
                      streaming=False, data_type=memcpy_dtype_32, nonblock=False, order=memcpy_order)
    time_ref_1d = np.zeros(P * P * 2, dtype=np.float32)
    runner.memcpy_d2h(time_ref_1d, symbol_time_ref, 0, 0, P, P, 2,
                      streaming=False, data_type=memcpy_dtype_32, nonblock=False, order=memcpy_order)

print("Copied back result.")
print("y calculated: ", y[:8], "...")
print("y expected:   ", y_expected[:8], "...")
np.testing.assert_allclose(y, y_expected, atol=0.5, rtol=0)
print("SUCCESS")

# Decode timing (same pattern as waferllm-gemv)
time_memcpy_hwl = time_memcpy_1d.reshape(P, P, 3)
time_ref_hwl    = time_ref_1d.reshape(P, P, 2)

time_start = np.zeros((P, P), dtype=int)
time_end   = np.zeros((P, P), dtype=int)
word       = np.zeros(3, dtype=np.uint16)
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

time_ref = np.zeros((P, P), dtype=int)
for w in range(P):
    for h in range(P):
        hex_t0 = int(float_to_hex(time_ref_hwl[h, w, 0]), base=16)
        hex_t1 = int(float_to_hex(time_ref_hwl[h, w, 1]), base=16)
        word[0] = hex_t0 & 0x0000ffff
        word[1] = (hex_t0 >> 16) & 0x0000ffff
        word[2] = hex_t1 & 0x0000ffff
        time_ref[h, w] = make_u48(word)

for _py in range(P):
    for _px in range(P):
        time_ref[_py, _px] -= (_px + _py)

time_start -= time_ref
time_end   -= time_ref

max_cycles = int(time_end.max() - time_start.min())
freq_ghz   = 1.1
time_ms    = max_cycles / (freq_ghz * 1e6)

print(f"\nP: {P}, M: {M}, N: {N}")
print(f"Max Cycle count: {max_cycles}")
print(f"Time: {time_ms:.6f} ms")
