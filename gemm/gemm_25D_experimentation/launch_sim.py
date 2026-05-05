"""2.5D mini-MeshGEMM Phase 1 launcher.

Per-layer Cannon GEMM on a P_phys x P_phys fabric, partitioned into
c = sqrt_c^2 layers of size q x q (q = P_phys / sqrt_c).

Phase 1 stops after the per-layer Cannon completes — each PE holds the
partial sum for ITS layer's K-slab.  The host then sums the c partials
at the same (lx, ly) to recover the full C and compares to numpy matmul.

Distribution rules:
    Mt       = M / q              (M-rows duplicated across lz_y)
    Nt       = N / q              (N-cols duplicated across lz_x)
    Kt_layer = K / (c * q)        (K split first across c layers, then q within)

PE at (px, py)  ->  lx = px % q,  ly = py % q,
                    lz_x = px // q,  lz_y = py // q,
                    lz = lz_y * sqrt_c + lz_x.

Layer lz holds K-slab K[lz*K_layer : (lz+1)*K_layer], where K_layer = K/c.
"""

import argparse
import struct

import numpy as np

from cerebras.sdk.runtime.sdkruntimepybind import (
    MemcpyDataType,
    MemcpyOrder,
    SdkRuntime,
)
from cerebras.sdk.sdk_utils import input_array_to_u32, memcpy_view


def parse_args():
    p = argparse.ArgumentParser(description="2.5D mini-MeshGEMM Phase 1 launcher")
    p.add_argument("--P_phys", required=True, type=int)
    p.add_argument("--M", required=True, type=int)
    p.add_argument("--K", required=True, type=int)
    p.add_argument("--N", required=True, type=int)
    p.add_argument("--sqrt_c", required=True, type=int,
                   help="sqrt of the number of layers (Phase 1 supports sqrt_c=2)")
    return p.parse_args()


def float_to_hex(f):
    return hex(struct.unpack("<I", struct.pack("<f", f))[0])


def make_u48(words):
    return words[0] + (words[1] << 16) + (words[2] << 32)


def assignId(pc, P):
    """Cannon initial-alignment index permutation, copied from MeshGEMM."""
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


def build_layer_ind(q):
    """Per-layer Cannon B-permutation indices, q x q."""
    ind = np.zeros((q, q), dtype=int)
    for i in range(q):
        for j in range(q):
            if i == 0:
                ind[0, j] = j
            elif i == 1:
                _, ind[1, j] = assignId(ind[0, j], q)
            else:
                if (i - 1) % 2 == 0:
                    _, ind[i, j] = assignId(ind[i - 2, j], q)
                else:
                    ind[i, j], _ = assignId(ind[i - 2, j], q)
    return ind


def main():
    np.random.seed(2025)

    args = parse_args()
    P_phys = args.P_phys
    M, K, N = args.M, args.K, args.N
    sqrt_c = args.sqrt_c
    c = sqrt_c * sqrt_c
    q = P_phys // sqrt_c

    assert P_phys == sqrt_c * q, f"P_phys must equal sqrt_c*q (got {P_phys} vs {sqrt_c}*{q})"
    assert M % q == 0, f"M ({M}) must be divisible by q ({q})"
    assert N % q == 0, f"N ({N}) must be divisible by q ({q})"
    assert K % (c * q) == 0, f"K ({K}) must be divisible by c*q ({c*q})"

    Mt = M // q
    Nt = N // q
    K_layer = K // c
    Kt_layer = K_layer // q

    print(f"P_phys={P_phys}, sqrt_c={sqrt_c}, c={c}, q={q}")
    print(f"M={M}, K={K}, N={N}  ->  Mt={Mt}, Kt_layer={Kt_layer}, Nt={Nt}")
    print(f"K_layer (per-layer K slab) = {K_layer}")

    io_dtype = MemcpyDataType.MEMCPY_16BIT
    memcpy_order = MemcpyOrder.ROW_MAJOR

    tensor_X = np.random.rand(M, K).astype(np.float16)
    tensor_W = np.random.rand(K, N).astype(np.float16)

    # ---- Build per-layer Cannon B-permutation ----
    ind_layer = np.stack([build_layer_ind(q) for _ in range(c)], axis=0)  # (c, q, q)

    tensor_W_perm = np.zeros_like(tensor_W)
    for layer in range(c):
        K_slab = tensor_W[layer * K_layer : (layer + 1) * K_layer]      # (K_layer, N)
        slab_perm = np.zeros_like(K_slab)
        for i in range(q):                  # i = lx (K-row block)
            for j in range(q):              # j = ly (N-col block)
                t = ind_layer[layer][i, j]
                slab_perm[i * Kt_layer : (i + 1) * Kt_layer,
                          j * Nt : (j + 1) * Nt] = \
                    K_slab[t * Kt_layer : (t + 1) * Kt_layer,
                           j * Nt : (j + 1) * Nt]
        tensor_W_perm[layer * K_layer : (layer + 1) * K_layer] = slab_perm

    # ---- Distribute X and W into per-PE local tiles ----
    X3 = np.zeros((P_phys, P_phys, Mt * Kt_layer), dtype=np.float16)
    W3 = np.zeros((P_phys, P_phys, Kt_layer * Nt), dtype=np.float16)

    for px in range(P_phys):
        for py in range(P_phys):
            lx = px % q
            ly = py % q
            lz_x = px // q
            lz_y = py // q
            lz = lz_y * sqrt_c + lz_x

            x_tile = tensor_X[ly * Mt : (ly + 1) * Mt,
                              lz * K_layer + lx * Kt_layer :
                              lz * K_layer + (lx + 1) * Kt_layer]
            # column-major Mt x Kt_layer: column k contiguous in memory
            X3[py, px, :] = x_tile.flatten(order="F")

            w_tile = tensor_W_perm[lz * K_layer + ly * Kt_layer :
                                   lz * K_layer + (ly + 1) * Kt_layer,
                                   lx * Nt : (lx + 1) * Nt]
            # row-major Kt_layer x Nt: row k contiguous
            W3[py, px, :] = w_tile.flatten(order="C")

    # ---- Launch ----
    runner = SdkRuntime("out")
    runner.load()
    runner.run()

    sym_X = runner.get_id("X")
    sym_W = runner.get_id("W")
    sym_res = runner.get_id("res")
    sym_time_memcpy = runner.get_id("time_memcpy")
    sym_time_ref = runner.get_id("time_ref")

    X_u32 = input_array_to_u32(X3.ravel(), 1, 1)
    runner.memcpy_h2d(sym_X, X_u32, 0, 0, P_phys, P_phys, Mt * Kt_layer,
                      streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False)

    W_u32 = input_array_to_u32(W3.ravel(), 1, 1)
    runner.memcpy_h2d(sym_W, W_u32, 0, 0, P_phys, P_phys, Kt_layer * Nt,
                      streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False)

    runner.launch("init_task", nonblock=False)
    total_warmup_times, total_repeat_times = 0, 1
    runner.launch("meshgemm_host", np.int16(total_warmup_times),
                  np.int16(total_repeat_times), nonblock=False)

    # Phase 2 reduces partials INTO layer 0 only.  D2H pulls just the q × q
    # top-left block (layer 0) — layers 1..c-1 hold stale data we don't read.
    res3_1d_u32 = np.zeros(q * q * Mt * Nt, dtype=np.uint32)
    runner.memcpy_d2h(res3_1d_u32, sym_res, 0, 0, q, q, Mt * Nt,
                      streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False)
    res3_1d_fp16 = memcpy_view(res3_1d_u32, np.dtype(np.float16))
    # res tile is col-major Mt x Nt, so its (Mt*Nt) flat layout is (Nt, Mt) row-major.
    res3 = res3_1d_fp16.reshape((q, q, Nt, Mt))

    # ---- Timing run ----
    runner.launch("init_task", nonblock=False)
    total_warmup_times, total_repeat_times = 1, 5
    runner.launch("meshgemm_host", np.int16(total_warmup_times),
                  np.int16(total_repeat_times), nonblock=False)

    time_memcpy_1d = np.zeros(P_phys * P_phys * 3, dtype=np.float32)
    runner.memcpy_d2h(time_memcpy_1d, sym_time_memcpy, 0, 0, P_phys, P_phys, 3,
                      streaming=False, order=MemcpyOrder.ROW_MAJOR,
                      data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)
    time_memcpy_hwl = time_memcpy_1d.reshape((P_phys, P_phys, 3), order="C")

    time_ref_1d = np.zeros(P_phys * P_phys * 2, dtype=np.float32)
    runner.memcpy_d2h(time_ref_1d, sym_time_ref, 0, 0, P_phys, P_phys, 2,
                      streaming=False, order=MemcpyOrder.ROW_MAJOR,
                      data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)
    time_ref_hwl = time_ref_1d.reshape((P_phys, P_phys, 2), order="C")

    runner.stop()

    # res3 is the layer-0 q × q block (lz_x == 0, lz_y == 0). Stitch the
    # column-major Mt × Nt tiles into the full M × N matrix.
    res_full = np.zeros((q, q, Mt, Nt), dtype=np.float16)
    for lx in range(q):
        for ly in range(q):
            res_full[ly, lx] = res3[ly, lx].T  # (Mt, Nt)
    res = res_full.transpose(0, 2, 1, 3).reshape(q * Mt, q * Nt)

    expected = np.matmul(tensor_X.astype(np.float32),
                         tensor_W.astype(np.float32)).astype(np.float16)

    print("\n--- Correctness ---")
    print(f"Expected[0,:8] = {expected[0, :8]}")
    print(f"Actual  [0,:8] = {res[0, :8]}")
    abs_err = np.abs(res.astype(np.float32) - expected.astype(np.float32))
    rel_err = abs_err / (np.abs(expected.astype(np.float32)) + 1e-6)
    print(f"max abs err = {abs_err.max():.4f}, mean abs err = {abs_err.mean():.4f}")
    print(f"max rel err = {rel_err.max():.4f}, mean rel err = {rel_err.mean():.4f}")

    np.testing.assert_allclose(expected, res, rtol=0.5, atol=0)
    print("PASSED: np.testing.assert_allclose(expected, res, rtol=0.5, atol=0)")

    # ---- Cycle counting (max across timed PEs) ----
    time_start = np.zeros((P_phys, P_phys), dtype=int)
    time_end = np.zeros((P_phys, P_phys), dtype=int)
    word = np.zeros(3, dtype=np.uint16)
    for w in range(P_phys):
        for h in range(P_phys):
            t0 = int(float_to_hex(time_memcpy_hwl[h, w, 0]), 16)
            t1 = int(float_to_hex(time_memcpy_hwl[h, w, 1]), 16)
            t2 = int(float_to_hex(time_memcpy_hwl[h, w, 2]), 16)
            word[0] = t0 & 0xFFFF
            word[1] = (t0 >> 16) & 0xFFFF
            word[2] = t1 & 0xFFFF
            time_start[h, w] = make_u48(word)
            word[0] = (t1 >> 16) & 0xFFFF
            word[1] = t2 & 0xFFFF
            word[2] = (t2 >> 16) & 0xFFFF
            time_end[h, w] = make_u48(word)

    time_ref = np.zeros((P_phys, P_phys), dtype=int)
    word = np.zeros(3, dtype=np.uint16)
    for w in range(P_phys):
        for h in range(P_phys):
            t0 = int(float_to_hex(time_ref_hwl[h, w, 0]), 16)
            t1 = int(float_to_hex(time_ref_hwl[h, w, 1]), 16)
            word[0] = t0 & 0xFFFF
            word[1] = (t0 >> 16) & 0xFFFF
            word[2] = t1 & 0xFFFF
            time_ref[h, w] = make_u48(word)

    for py in range(P_phys):
        for px in range(P_phys):
            time_ref[py, px] -= (px + py)
    time_start -= time_ref
    time_end -= time_ref

    print(f"\n--- Timing (over {total_repeat_times} timed iters) ---")
    print(f"Mean cycle count: {(time_end - time_start).mean() / total_repeat_times:.1f}")
    print(f"Max  cycle count: {(time_end.max() - time_start.min()) / total_repeat_times:.1f}")


if __name__ == "__main__":
    main()
