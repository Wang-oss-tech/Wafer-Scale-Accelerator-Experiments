# 2.5D GEMM Experiment on Cerebras WSE (Memory-Based, Extra-Memory Interpretation)

## Goal

Implement a 2.5D GEMM experiment on WSE-3 in a new directory `gemm_25D_experiment` under `/home/william_wang_yale/R_2.6.0/csl-examples/experiments/gemm/`.

The "2.5D" here means: keep the same P×P PE grid as the baseline, but each PE pre-loads c=2 A-tile and B-tile pairs directly from the host into local memory (48KB per PE). The first c=2 SUMMA steps are performed using only locally held tiles — no broadcasts needed. The remaining P-c=2 SUMMA steps use a custom `@fmovh`-based two-tree broadcast (replacing the existing tile_config `@fmovs` approach), inspired by `two_tree_allreduce_y` in `experiments/gemv/waferllm-gemv/src/comm_lib/comm_pe.csl`.

## Starting Point

Duplicate `summa_manual_multicasting_pipelined_doubleColor-fp16_optimized` to get the validation framework (layout, fp16 tiles, timestamps, run.py). The copy must first run `./commands_wse3.sh` unchanged before any algorithm changes are made.

## Algorithm

Standard 2D SUMMA (baseline, P=4 steps):
- Each step k: column-k PEs broadcast A tile along row; row-k PEs broadcast B tile along column; all PEs compute C += A*B

2.5D extra-memory variant (c=2):
1. Host pre-loads c=2 A tiles and c=2 B tiles per PE via H2D memcpy
   - Each PE at (px, py) receives: A tiles for steps [px, px+1] mod P (the steps it owns locally) and B tiles for steps [py, py+1] mod P
   - These are the tiles that would normally arrive via broadcast
2. Each PE performs c=2 local GEMM steps using pre-loaded tiles — no inter-PE communication
3. For the remaining P-c=2 steps, custom two-tree broadcast using `@fmovh`/`@faddh` with `simd_mode = .{ .simd_max = true }`
   - A tile broadcast: row direction (x-axis) from column-k root
   - B tile broadcast: column direction (y-axis) from row-k root
   - Pattern mirrors the broadcast phase of `two_tree_allreduce_y`

## Custom Broadcast Design

Reference: `experiments/gemv/waferllm-gemv/src/comm_lib/comm_pe.csl`

The `two_tree_allreduce_y` function demonstrates:
- Using `@fmovh` to send fp16 wavelets to fabric
- Using `@faddh` to receive and accumulate fp16 from fabric
- `simd_mode = .{ .simd_max = true }` on all fabric DSDs
- Separate up/down colors for bidirectional tree communication
- WSE-3 `@initialize_queue` calls in comptime block

For GEMM, we need:
- **B broadcast (y-axis)**: broadcast from row-k root PE down/up its column — use the broadcast phase of `two_tree_allreduce_y`
- **A broadcast (x-axis)**: broadcast from column-k root PE left/right along its row — analogous x-axis version

## Requirements

1. **Directory**: Create `gemm_25D_experiment/` by copying `summa_manual_multicasting_pipelined_doubleColor-fp16_optimized/`
2. **Baseline**: Unmodified copy must pass `./commands_wse3.sh` (P=4, Mt=Kt=Nt=14)
3. **c=2 local GEMM**: PE holds 2 extra A+B tile pairs; first 2 SUMMA steps run locally
4. **Custom @fmovh broadcast**: Remaining 2 broadcast steps use two-tree @fmovh pattern
5. **run.py interface**: Match MeshGEMM run.py (`--P --M --K --N --warmup --repeats`, MEMCPY_16BIT)
6. **fabric-dims**: Unchanged at `11,6` (same P×P grid)
7. **Timestamps**: Preserve 48-bit TSC infrastructure; report cycle counts

## Design Notes

- `@fmovs` (packed u32) is replaced by `@fmovh` (native fp16) for all tile broadcast wavelet transfers
- All fabric DSDs use `simd_mode = .{ .simd_max = true }` for fp16 throughput
- WSE-3 requires explicit `@initialize_queue` calls in comptime for all queue/color pairs
- The double-color pipelining (even/odd) from the baseline may be simplified or removed since the local steps don't need it
- Debug: use `<simprint>` with `SINGULARITYENV_SIMFABRIC_DEBUG` commented out; check `sim.log`

## Success Criteria

- `./commands_wse3.sh` compiles and simulates without error
- `run.py --P 4 --M 56 --K 56 --N 56` prints "Correctness check PASSED"
- Each PE uses exactly c=2 pre-loaded tiles for 2 local GEMM steps
- Remaining 2 steps use `@fmovh`-based broadcast with no `@fmovs` calls for tile transfer
