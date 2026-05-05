import argparse
import numpy as np

from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime, MemcpyDataType, MemcpyOrder
from cerebras.sdk.sdk_utils import calculate_cycles


def parse_args():
    parser = argparse.ArgumentParser(description="WaferLLM Decode WSE-3 simulator runner")
    parser.add_argument("--name",      required=True,          help="Compiled artifact directory")
    parser.add_argument("--cmaddr",    default=None,           help="Appliance IP:port (omit for simulator)")
    parser.add_argument("--P",         required=True, type=int, help="PE grid size P×P")
    parser.add_argument("--bsz",       default=1,     type=int, help="Batch size (default 1)")
    parser.add_argument("--dim",       required=True, type=int, help="Model hidden dimension")
    parser.add_argument("--seq_len",   required=True, type=int, help="Sequence length")
    parser.add_argument("--ffn_dim",   required=True, type=int, help="FFN intermediate dimension")
    parser.add_argument("--head_dim",  required=True, type=int, help="Attention head dimension")
    parser.add_argument("--n_heads",   default=1,     type=int, help="Number of attention heads")
    parser.add_argument("--n_kv_heads",default=1,     type=int, help="Number of KV heads")
    parser.add_argument("--group_num", default=2,     type=int, help="Number of allreduce groups")
    parser.add_argument("--warmup",    default=0,     type=int, help="Warmup iterations (default 0)")
    parser.add_argument("--repeats",   default=1,     type=int, help="Timed iterations (default 1)")
    parser.add_argument("--layer_num", default=1,     type=int, help="Number of transformer layers (for full-model TPR)")
    return parser.parse_args()


def to_u32_16bit(arr):
    return arr.ravel().view(np.uint16).astype(np.uint32)


def main():
    args = parse_args()

    P           = args.P
    bsz         = args.bsz
    dim         = args.dim
    seq_len     = args.seq_len
    ffn_dim     = args.ffn_dim
    head_dim    = args.head_dim
    n_heads     = args.n_heads
    n_kv_heads  = args.n_kv_heads
    group_num   = args.group_num
    warmup      = args.warmup
    repeats     = args.repeats

    dim_p_pe      = dim // P
    pes_p_head    = P // n_heads
    pes_p_kv_head = P // n_kv_heads
    head_dim_p_pe = head_dim // P
    seq_len_p_pe  = seq_len // P
    ffn_dim_p_pe  = ffn_dim // P
    pe_num_p_group = P // group_num

    # dim_p_pe must be even for RoPE; clip if odd
    _dim_p_pe = (dim_p_pe // 2) * 2

    print(f"P={P} bsz={bsz} dim_p_pe={dim_p_pe} pes_p_head={pes_p_head} "
          f"pes_p_kv_head={pes_p_kv_head} head_dim_p_pe={head_dim_p_pe} "
          f"seq_len_p_pe={seq_len_p_pe} ffn_dim_p_pe={ffn_dim_p_pe} "
          f"pe_num_p_group={pe_num_p_group}")

    io_dtype     = MemcpyDataType.MEMCPY_16BIT
    memcpy_order = MemcpyOrder.ROW_MAJOR

    # ------------------------------------------------------------------
    # Build input tensors
    # ------------------------------------------------------------------
    X = np.random.rand(1, bsz * dim).astype(np.float16)
    tensor_X = np.tile(X.reshape(P, bsz * dim_p_pe), reps=(1, P))

    W = np.random.rand(1, dim).astype(np.float16)
    tensor_W = np.tile(W.reshape(P, dim_p_pe), reps=(1, P))

    tensor_q_weight = np.random.rand(dim, dim).astype(np.float16)
    tensor_k_weight = np.random.rand(dim, dim).astype(np.float16)
    tensor_v_weight = np.random.rand(dim, dim).astype(np.float16)

    freqs_sin = np.random.rand(1, P * _dim_p_pe // 2).astype(np.float16)
    tensor_freqs_sin = np.tile(freqs_sin.reshape(P, _dim_p_pe // 2), reps=(1, P))
    freqs_cos = np.random.rand(1, P * _dim_p_pe // 2).astype(np.float16)
    tensor_freqs_cos = np.tile(freqs_cos.reshape(P, _dim_p_pe // 2), reps=(1, P))

    tensor_XKCache = np.random.rand(dim, seq_len).astype(np.float16)
    tensor_XVCache = np.random.rand(seq_len, dim).astype(np.float16)

    tensor_o_weight   = np.random.rand(dim, dim).astype(np.float16)
    tensor_up_weight  = np.random.rand(dim, ffn_dim).astype(np.float16)
    tensor_gate_weight= np.random.rand(dim, ffn_dim).astype(np.float16)
    tensor_down_weight= np.random.rand(ffn_dim, dim).astype(np.float16)

    # ------------------------------------------------------------------
    # Tile weight matrices: (dim, dim) -> (P, P, dim_p_pe*dim_p_pe)
    # ------------------------------------------------------------------
    def tile_weight_dim_dim(w):
        r = w.reshape(P, dim_p_pe, P, dim_p_pe).transpose(0, 2, 1, 3).reshape(P, P, dim_p_pe * dim_p_pe)
        return r

    def tile_weight_dim_ffn(w):
        r = w.reshape(P, dim_p_pe, P, ffn_dim_p_pe).transpose(0, 2, 1, 3).reshape(P, P, dim_p_pe * ffn_dim_p_pe)
        return r

    def tile_weight_ffn_dim(w):
        r = w.reshape(P, ffn_dim_p_pe, P, dim_p_pe).transpose(0, 2, 1, 3).reshape(P, P, ffn_dim_p_pe * dim_p_pe)
        return r

    def tile_xkcache(w):
        r = w.reshape(P, dim_p_pe, P, seq_len_p_pe).transpose(0, 2, 1, 3).reshape(P, P, dim_p_pe * seq_len_p_pe)
        return r

    def tile_xvcache(w):
        r = w.reshape(P, seq_len_p_pe, P, dim_p_pe).transpose(0, 2, 1, 3).reshape(P, P, seq_len_p_pe * dim_p_pe)
        return r

    runner = SdkRuntime(args.name, cmaddr=args.cmaddr)
    runner.load()
    runner.run()

    sym_X         = runner.get_id("X")
    sym_W         = runner.get_id("W")
    sym_Q_weight  = runner.get_id("Q_weight")
    sym_K_weight  = runner.get_id("K_weight")
    sym_V_weight  = runner.get_id("V_weight")
    sym_freqs_sin = runner.get_id("freqs_sin")
    sym_freqs_cos = runner.get_id("freqs_cos")
    sym_XKCache   = runner.get_id("XKCache")
    sym_XVCache   = runner.get_id("XVCache")
    sym_O_weight  = runner.get_id("O_weight")
    sym_UP_weight = runner.get_id("UP_weight")
    sym_GATE_weight = runner.get_id("GATE_weight")
    sym_DOWN_weight = runner.get_id("DOWN_weight")
    sym_timer_buf = runner.get_id("timer_buf")
    sym_time_ref  = runner.get_id("time_ref")
    sym_debug     = runner.get_id("debug")

    # ------------------------------------------------------------------
    # H2D memcpy
    # ------------------------------------------------------------------
    runner.memcpy_h2d(sym_X, to_u32_16bit(tensor_X),
                      0, 0, P, P, bsz * dim_p_pe,
                      streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False)

    runner.memcpy_h2d(sym_W, to_u32_16bit(tensor_W),
                      0, 0, P, P, dim_p_pe,
                      streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False)

    runner.memcpy_h2d(sym_Q_weight, to_u32_16bit(tile_weight_dim_dim(tensor_q_weight)),
                      0, 0, P, P, dim_p_pe * dim_p_pe,
                      streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False)

    runner.memcpy_h2d(sym_K_weight, to_u32_16bit(tile_weight_dim_dim(tensor_k_weight)),
                      0, 0, P, P, dim_p_pe * dim_p_pe,
                      streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False)

    runner.memcpy_h2d(sym_V_weight, to_u32_16bit(tile_weight_dim_dim(tensor_v_weight)),
                      0, 0, P, P, dim_p_pe * dim_p_pe,
                      streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False)

    runner.memcpy_h2d(sym_freqs_sin, to_u32_16bit(tensor_freqs_sin),
                      0, 0, P, P, _dim_p_pe // 2,
                      streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False)

    runner.memcpy_h2d(sym_freqs_cos, to_u32_16bit(tensor_freqs_cos),
                      0, 0, P, P, _dim_p_pe // 2,
                      streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False)

    runner.memcpy_h2d(sym_XKCache, to_u32_16bit(tile_xkcache(tensor_XKCache)),
                      0, 0, P, P, dim_p_pe * seq_len_p_pe,
                      streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False)

    runner.memcpy_h2d(sym_XVCache, to_u32_16bit(tile_xvcache(tensor_XVCache)),
                      0, 0, P, P, seq_len_p_pe * dim_p_pe,
                      streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False)

    runner.memcpy_h2d(sym_O_weight, to_u32_16bit(tile_weight_dim_dim(tensor_o_weight)),
                      0, 0, P, P, dim_p_pe * dim_p_pe,
                      streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False)

    runner.memcpy_h2d(sym_UP_weight, to_u32_16bit(tile_weight_dim_ffn(tensor_up_weight)),
                      0, 0, P, P, dim_p_pe * ffn_dim_p_pe,
                      streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False)

    runner.memcpy_h2d(sym_GATE_weight, to_u32_16bit(tile_weight_dim_ffn(tensor_gate_weight)),
                      0, 0, P, P, dim_p_pe * ffn_dim_p_pe,
                      streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False)

    runner.memcpy_h2d(sym_DOWN_weight, to_u32_16bit(tile_weight_ffn_dim(tensor_down_weight)),
                      0, 0, P, P, ffn_dim_p_pe * dim_p_pe,
                      streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False)

    # ------------------------------------------------------------------
    # Launch kernel
    # ------------------------------------------------------------------
    runner.launch("init_task", nonblock=False)
    runner.launch("decode_host", np.int16(warmup), np.int16(repeats), nonblock=False)

    # ------------------------------------------------------------------
    # D2H memcpy
    # ------------------------------------------------------------------
    debug_u32 = np.zeros(P * P * bsz * dim_p_pe, dtype=np.uint32)
    runner.memcpy_d2h(debug_u32, sym_debug,
                      0, 0, P, P, bsz * dim_p_pe,
                      streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False)

    timer_u32 = np.zeros(P * P * 3, dtype=np.uint32)
    runner.memcpy_d2h(timer_u32, sym_timer_buf,
                      0, 0, P, P, 3,
                      streaming=False, data_type=MemcpyDataType.MEMCPY_32BIT,
                      order=MemcpyOrder.ROW_MAJOR, nonblock=False)

    runner.stop()

    # ------------------------------------------------------------------
    # Timing
    # ------------------------------------------------------------------
    timer_hwl = timer_u32.view(np.float32).reshape(P, P, 3)
    cycles = np.zeros((P, P))
    for py in range(P):
        for px in range(P):
            cycles[py, px] = calculate_cycles(timer_hwl[py, px, :])

    cycles_per_step = cycles / repeats
    mean_cycles = cycles_per_step.mean()
    max_cycles  = cycles_per_step.max()
    freq_ghz = 1.1
    mean_ms = mean_cycles / (freq_ghz * 1e6)
    layer_num = args.layer_num

    print(f"Warmup={warmup} Repeats={repeats}")
    print(f"Mean cycle count: {mean_cycles:.1f}")
    print(f"Max  cycle count: {max_cycles:.1f}")
    print(f"Mean time: {mean_ms:.4f} ms")
    print(f"Decode throughput per request: {1.0 / (mean_ms * 1e-3 * layer_num):.1f} tokens/s")


if __name__ == "__main__":
    main()
