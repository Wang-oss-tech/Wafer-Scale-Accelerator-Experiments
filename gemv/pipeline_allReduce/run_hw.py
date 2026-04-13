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


def cast_tensor_u32(tensor):
    return np.uint32(tensor.view(np.uint16))


def parse_args():
    parser = argparse.ArgumentParser(description="pipeline-allreduce GEMV hardware runner")
    parser.add_argument("--P",         required=True, type=int, help="PE grid size P x P")
    parser.add_argument("--M",         required=True, type=int, help="Input dimension of x(1xM) and W(MxN)")
    parser.add_argument("--N",         required=True, type=int, help="Output dimension of W(MxN) and y(1xN)")
    parser.add_argument("--warmup",    default=10, type=int, help="Warmup runs (default 10)")
    parser.add_argument("--repeats",   default=100, type=int, help="Timed runs (default 100)")
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
X = np.random.rand(1, M).astype(np.float16)
tensor_X = np.tile(X.reshape(P, M // P), reps=(1, P))
W = np.random.rand(M, N).astype(np.float16)
y_expected = (X.astype(np.float32) @ W.astype(np.float32)).astype(np.float16).reshape(N)

kernel_rows = P
kernel_cols = P
per_pe_rows = M // P
per_pe_cols = N // P

X_u32 = cast_tensor_u32(tensor_X.ravel())
W1 = W.reshape(kernel_rows, per_pe_rows, kernel_cols, per_pe_cols)
W2 = W1.transpose(0, 2, 1, 3)
W3 = W2.reshape(kernel_rows, kernel_cols, per_pe_rows * per_pe_cols)
W_u32 = cast_tensor_u32(W3.ravel())

print("Start running...", flush=True)
with SdkRuntime(artifact_id, simulator=args.simulator, disable_version_check=True) as runner:

    symbol_X           = runner.get_id("X")
    symbol_W           = runner.get_id("W")
    symbol_res         = runner.get_id("res")
    symbol_time_memcpy = runner.get_id("time_memcpy")
    symbol_time_ref    = runner.get_id("time_ref")

    runner.memcpy_h2d(symbol_X, X_u32, 0, 0, kernel_cols, kernel_rows, per_pe_rows,
                      streaming=False, data_type=memcpy_dtype_16, nonblock=False, order=memcpy_order)
    runner.memcpy_h2d(symbol_W, W_u32, 0, 0, kernel_cols, kernel_rows, per_pe_rows * per_pe_cols,
                      streaming=False, data_type=memcpy_dtype_16, nonblock=False, order=memcpy_order)

    runner.launch("init_task", nonblock=False)
    runner.launch("pipeline_allreduce_host", np.int16(args.warmup), np.int16(args.repeats), nonblock=False)

    res_u32 = np.zeros(P * N, dtype=np.uint32)
    runner.memcpy_d2h(res_u32, symbol_res, 0, 0, kernel_cols, kernel_rows, per_pe_cols,
                      streaming=False, data_type=memcpy_dtype_16, nonblock=False, order=memcpy_order)
    res = sdk_utils.memcpy_view(res_u32, np.dtype(np.float16)).reshape(P, N)

    time_memcpy_1d = np.zeros(P * P * 3, dtype=np.float32)
    runner.memcpy_d2h(time_memcpy_1d, symbol_time_memcpy, 0, 0, P, P, 3,
                      streaming=False, data_type=memcpy_dtype_32, nonblock=False, order=memcpy_order)
    time_ref_1d = np.zeros(P * P * 2, dtype=np.float32)
    runner.memcpy_d2h(time_ref_1d, symbol_time_ref, 0, 0, P, P, 2,
                      streaming=False, data_type=memcpy_dtype_32, nonblock=False, order=memcpy_order)

print("Copied back result.")
print("res calculated (row 0): ", res[0, :8], "...")
print("y expected:             ", y_expected[:8], "...")
expected_rows = np.tile(y_expected.reshape(1, N), reps=(P, 1))
np.testing.assert_allclose(res, expected_rows, atol=0.5, rtol=0)
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

print(f"\nRepeat count: {args.repeats}")
print(f"P: {P}, M: {M}, N: {N}")
print(f"Mean cycle count: {np.mean(time_end - time_start) / args.repeats}")
print(f"Max Cycle count: {max_cycles / args.repeats}")
print(f"Time: {time_ms:.6f} ms")
