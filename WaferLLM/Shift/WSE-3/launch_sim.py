import random
import numpy as np
import argparse
import struct

from cerebras.sdk.sdk_utils import input_array_to_u32, memcpy_view
from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime
from cerebras.sdk.runtime.sdkruntimepybind import MemcpyDataType, MemcpyOrder

def parse_args():
    
    parser = argparse.ArgumentParser(description="MeshGEMV on simulator")
    parser.add_argument("--P", required=True, type=int, help="PEs rectangle size: P x P")
    parser.add_argument("--L", required=True, type=int, help="Shift distance: L")
    parser.add_argument("--M", required=True, type=int, help="Left vector dimension: 1 x M")
    parser.add_argument("--N", required=True, type=int, help="Right matrix dimension: M x N")
    
    args = parser.parse_args()
    return args

def float_to_hex(f):
    return hex(struct.unpack('<I', struct.pack('<f', f))[0])

def make_u48(words):
    return words[0] + (words[1] << 16) + (words[2] << 32)

def main():
    random.seed(2025)
    
    args = parse_args()
    
    P = args.P
    L = args.L
    
    M = args.M
    N = args.N
    
    Mt = M // P
    Nt = N // P
    
    io_dtype = MemcpyDataType.MEMCPY_16BIT
    memcpy_order = MemcpyOrder.ROW_MAJOR

    tensor_W = np.random.rand(M, N).astype(np.float16)
    
    runner = SdkRuntime("out")
    runner.load()
    runner.run()

    sym_W = runner.get_id("W")
    sym_res = runner.get_id("res")
    
    symbol_time_memcpy = runner.get_id("time_memcpy")
    
    W1 = tensor_W.reshape(P, Mt, P, Nt)
    W2 = W1.transpose(0, 2, 1, 3)
    W3 = W2.reshape(P, P, Mt*Nt)
    W_u32 = input_array_to_u32(W3.ravel(), 1, 1)
    runner.memcpy_h2d(sym_W, W_u32, 0, 0, P, P, Mt*Nt, \
                      streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False)
    
    runner.launch('init_task', nonblock=False)
    total_warmup_times, total_repeat_times = 1, 5
    runner.launch('shift_host', np.int16(total_warmup_times), np.int16(total_repeat_times), nonblock=False)
    
    res3_1d_u32 = np.zeros(M*N, dtype=np.uint32)
    runner.memcpy_d2h(res3_1d_u32, sym_res, L, 0, P, P, Mt*Nt, \
                      streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False)
    res3_1d_fp16 = memcpy_view(res3_1d_u32, np.dtype(np.float16))
    res3 = res3_1d_fp16.reshape((P, P, Mt, Nt))
    res2 = res3.transpose(0, 2, 1, 3)
    res = res2.reshape(M, N)
    
    src_time_memcpy_1d_f32 = np.zeros(P*P*3, dtype=np.float32)
    runner.memcpy_d2h(src_time_memcpy_1d_f32, symbol_time_memcpy, 0, 0, P, P, 3, streaming=False,
                    order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)
    src_time_memcpy_hwl = np.reshape(src_time_memcpy_1d_f32, (P, P, 3), order='C')
    
    des_time_memcpy_1d_f32 = np.zeros(P*P*3, dtype=np.float32)
    runner.memcpy_d2h(des_time_memcpy_1d_f32, symbol_time_memcpy, L, 0, P, P, 3, streaming=False,
                    order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)
    des_time_memcpy_hwl = np.reshape(des_time_memcpy_1d_f32, (P, P, 3), order='C')
    
    runner.stop()
    
    src_time_start = np.zeros((P, P)).astype(int)
    src_time_end = np.zeros((P, P)).astype(int)
    word = np.zeros(3).astype(np.uint16)
    for w in range(P):
        for h in range(P):
            hex_t0 = int(float_to_hex(src_time_memcpy_hwl[(h, w, 0)]), base=16)
            hex_t1 = int(float_to_hex(src_time_memcpy_hwl[(h, w, 1)]), base=16)
            hex_t2 = int(float_to_hex(src_time_memcpy_hwl[(h, w, 2)]), base=16)
            word[0] = hex_t0 & 0x0000ffff
            word[1] = (hex_t0 >> 16) & 0x0000ffff
            word[2] = hex_t1 & 0x0000ffff
            src_time_start[(h, w)] = make_u48(word)
            word[0] = (hex_t1 >> 16) & 0x0000ffff
            word[1] = hex_t2 & 0x0000ffff
            word[2] = (hex_t2 >> 16) & 0x0000ffff
            src_time_end[(h, w)] = make_u48(word)
            
    des_time_start = np.zeros((P, P)).astype(int)
    des_time_end = np.zeros((P, P)).astype(int)
    word = np.zeros(3).astype(np.uint16)
    for w in range(P):
        for h in range(P):
            hex_t0 = int(float_to_hex(des_time_memcpy_hwl[(h, w, 0)]), base=16)
            hex_t1 = int(float_to_hex(des_time_memcpy_hwl[(h, w, 1)]), base=16)
            hex_t2 = int(float_to_hex(des_time_memcpy_hwl[(h, w, 2)]), base=16)
            word[0] = hex_t0 & 0x0000ffff
            word[1] = (hex_t0 >> 16) & 0x0000ffff
            word[2] = hex_t1 & 0x0000ffff
            des_time_start[(h, w)] = make_u48(word)
            word[0] = (hex_t1 >> 16) & 0x0000ffff
            word[1] = hex_t2 & 0x0000ffff
            word[2] = (hex_t2 >> 16) & 0x0000ffff
            des_time_end[(h, w)] = make_u48(word)
            
    print("Expected result:")
    print(tensor_W)
    print("Actual result:")
    print(res)
    
    print(f"\nRepeat count: {total_repeat_times}")
    print(f"P: {P}, L: {L}, M: {M}, N: {N}")
    print(f"Mean cycle count: {np.mean(des_time_end - src_time_start)/total_repeat_times}")
    
if __name__ == "__main__":
    main()