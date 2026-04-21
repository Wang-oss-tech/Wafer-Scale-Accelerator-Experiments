# 2.5D GEMM Experiment on Cerebras WSE-3 (Extra-Memory Variant)

## Goal Description

Create `gemm_25D_experiment/` implementing a memory-based 2.5D GEMM on WSE-3. The same `P×P` grid as the baseline is kept. Each PE pre-loads c=2 extra A+B tile pairs from the host into its 48KB local memory, enabling the first c=2 SUMMA steps to run with zero inter-PE communication. The remaining P-c=2 steps use a custom `@fmovh`/`@faddh` two-tree broadcast (replacing the existing tile_config + `@fmovs` approach), modeled on the `two_tree_allreduce_y` pattern in `experiments/gemv/waferllm-gemv/src/comm_lib/comm_pe.csl`. The `run.py` host interface matches MeshGEMM (`--P --M --K --N --warmup --repeats`).

---

## Acceptance Criteria

- **AC-1**: `gemm_25D_experiment/` exists with baseline source files; unmodified `./commands_wse3.sh` passes.
  - Positive Tests (expected to PASS):
    - `./commands_wse3.sh` exits 0 and `run.py` prints "Correctness check PASSED" on the unmodified copy.
    - All baseline files (`layout.csl`, `pe.csl`, `run.py`, `commands_wse3.sh`) present.
  - Negative Tests (expected to FAIL):
    - Deliberate CSL syntax error in `layout.csl` causes `cslc` to fail.

- **AC-2**: PE tile layout holds c=2 A+B tile pairs; first 2 SUMMA steps run locally without any broadcast.
  - Positive Tests (expected to PASS):
    - Simprint log shows PE completing 2 GEMM steps before any fabric send/receive for A or B tiles.
    - `grep` on `pe.csl` finds arrays `A_local[2 * Mt * Kt]` and `B_local[2 * Kt * Nt]` (or equivalent double-tile buffers).
  - Negative Tests (expected to FAIL):
    - A version without pre-loaded tiles that attempts a broadcast for all P steps produces identical output but simprint shows 4 broadcast events instead of 2.

- **AC-3**: Remaining P-c=2 broadcast steps use `@fmovh`/`@faddh` two-tree broadcast; no `@fmovs` calls for tile transfer.
  - Positive Tests (expected to PASS):
    - `grep @fmovh pe.csl` finds broadcast send calls; `grep @faddh pe.csl` finds receive-accumulate calls.
    - `grep @fmovs pe.csl` returns empty (no packed-u32 tile transfers).
    - All fabric DSDs in the broadcast path include `simd_mode = .{ .simd_max = true }`.
  - Negative Tests (expected to FAIL):
    - Removing `simd_mode` from fabric DSDs compiles but produces measurably lower throughput in timing output.

- **AC-4**: End-to-end correctness: `run.py` numpy check passes for C = X @ W.
  - Positive Tests (expected to PASS):
    - `cs_python run.py --name out --P 4 --M 56 --K 56 --N 56 --warmup 0 --repeats 1` prints "Correctness check PASSED".
  - Negative Tests (expected to FAIL):
    - Skipping the local GEMM steps (only doing the 2 broadcast steps) produces wrong C; numpy check fails.
    - Using wrong tile indices for pre-loaded steps produces wrong C; numpy check fails.

- **AC-5**: `run.py` interface matches MeshGEMM: `--P --M --K --N --warmup --repeats`, `MEMCPY_16BIT`, cycle-count reporting.
  - Positive Tests (expected to PASS):
    - `cs_python run.py --name out --P 4 --M 56 --K 56 --N 56 --warmup 0 --repeats 1` parses correctly.
    - Output includes mean/max cycle count and ms estimate.
  - Negative Tests (expected to FAIL):
    - Passing `--Mt 14` raises argparse error (old interface rejected).
    - Passing `--M 55` (not divisible by P=4) raises reshape error before any memcpy.

---

## Path Boundaries

### Upper Bound (Maximum Acceptable Scope)
Full 2.5D extra-memory GEMM with c=2 pre-loaded tile steps, custom two-tree `@fmovh` broadcast for remaining steps, simprint debug blocks (commented out), preserved 48-bit TSC timestamps, and MeshGEMM-compatible run.py.

### Lower Bound (Minimum Acceptable Scope)
Correct 2.5D GEMM passing numpy validation with c=2 local steps and `@fmovh` for the 2 broadcast steps. Cycle-count output present. Simprint blocks may be commented out.

### Allowed Choices
- Can use: same `P×P` grid (`@set_rectangle(P, P)`) and `--fabric-dims=11,6` unchanged
- Can use: separate A-local and B-local buffer arrays (size `c * Mt * Kt` each)
- Can use: two-tree broadcast for y-axis (B tiles) adapted directly from `two_tree_allreduce_y`; x-axis (A tiles) as analogous mirror
- Can use: simplified or removed double-color even/odd pipelining (local steps don't need it)
- Cannot use: `@fmovs` for any tile wavelet transfers — must be `@fmovh`
- Cannot use: tile_config manual color register writes for the broadcast steps
- Cannot use: old `--Mt/--Kt/--Nt` argument interface in run.py

---

## Feasibility Hints and Suggestions

> **Note**: Conceptual suggestions only — not prescriptive requirements.

### Conceptual Approach

**PE tile layout**:
```csl
var A_local: [2 * _A_pad]f16 = @zeros([2 * _A_pad]f16);  // pre-loaded for 2 local steps
var B_local: [2 * _B_pad]f16 = @zeros([2 * _B_pad]f16);
var A_recv:  [_A_pad]f16 = @zeros([_A_pad]f16);           // broadcast receive buffer
var B_recv:  [_B_pad]f16 = @zeros([_B_pad]f16);
```

**SUMMA loop (P=4, c=2)**:
```
// Steps 0,1: local — no broadcast
for local_step in [0, 1]:
    do_gemm(A_local[local_step], B_local[local_step])

// Steps 2,3: custom @fmovh two-tree broadcast
for broadcast_step in [2, 3]:
    broadcast_A_fmovh(step=broadcast_step, recv=A_recv)
    broadcast_B_fmovh(step=broadcast_step, recv=B_recv)
    do_gemm(A_recv, B_recv)
```

**custom @fmovh broadcast (y-axis for B, x-axis for A)**:
Adapted from `two_tree_allreduce_y` in `comm_pe.csl`:
- Fabric DSDs: `simd_mode = .{ .simd_max = true }`, extent in f16 elements
- Row-k root sends B tile down/up column using `@fmovh(fab_out, mem_dsr)`
- Non-root PEs receive using `@fmovh(mem_dsd, fab_in)` (no accumulation — pure broadcast)
- WSE-3: `@initialize_queue` in comptime for all queue/color pairs

**Host pre-loading (run.py)**:
- For local step 0: PE(px, py) needs A tile from column px (its own) and B tile from row py (its own) — these are the baseline tiles
- For local step 1: PE(px, py) needs A tile from column `(px+1) % P` and B tile from row `(py+1) % P`
- Distribute both sets via H2D before calling `main`

### Relevant References
- `summa_manual_multicasting_pipelined_doubleColor-fp16_optimized/pe.csl` — baseline to modify
- `experiments/gemv/waferllm-gemv/src/comm_lib/comm_pe.csl` — `two_tree_allreduce_y` with `@fmovh`/`@faddh` and `simd_mode`
- `experiments/gemv/waferllm-gemv/src/comm_lib/comm_layout.csl` — color assignment pattern (up/down pairs, even/odd for alternating)
- `WaferLLM/MeshGEMM/WSE-3/run.py` — target run.py interface and MEMCPY_16BIT pattern

---

## Dependencies and Sequence

### Milestones

1. **Baseline Duplication**: Working copy of fp16 baseline.
   - Step A: Copy source files to `gemm_25D_experiment/`
   - Step B: Verify `./commands_wse3.sh` passes unchanged

2. **PE Tile Layout**: Extend pe.csl for c=2 pre-loaded tile buffers.
   - Step A: Add `A_local[2*_A_pad]`, `B_local[2*_B_pad]` arrays
   - Step B: Update local GEMM loop to run c=2 steps from pre-loaded tiles
   - Step C: Verify local-only mode compiles (run.py not yet updated)

3. **Custom @fmovh Broadcast**: Replace tile_config + @fmovs with two-tree @fmovh.
   - Step A: Define new colors/queues for x-axis (A) and y-axis (B) broadcasts
   - Step B: Implement y-axis B broadcast adapted from `two_tree_allreduce_y`
   - Step C: Implement x-axis A broadcast (analogous pattern)
   - Step D: Wire broadcast steps into SUMMA loop for steps [c, P-1]
   - Step E: Add `@initialize_queue` comptime blocks for WSE-3

4. **Host Program**: Rewrite run.py to MeshGEMM interface + 2.5D pre-loading.
   - Step A: Update argparse to `--P --M --K --N --warmup --repeats`
   - Step B: Distribute baseline tiles (step 0) + extra tiles (step 1) per PE
   - Step C: Collect C, reconstruct, validate with `rtol=0.5`
   - Step D: Port cycle-count unpacking and reporting

5. **End-to-End Validation**:
   - Step A: Run `./commands_wse3.sh`
   - Step B: Debug with simprint if needed (comment out `SINGULARITYENV_SIMFABRIC_DEBUG`)
   - Step C: Confirm "Correctness check PASSED"

---

## Task Breakdown

| Task ID | Description | Target AC | Tag | Depends On |
|---------|-------------|-----------|-----|------------|
| task1 | Copy baseline to `gemm_25D_experiment/`, verify `./commands_wse3.sh` passes | AC-1 | coding | - |
| task2 | Analyze how to map local-step tile indices to host distribution in run.py; verify memory fits in 48KB | AC-2, AC-5 | analyze | task1 |
| task3 | Extend `pe.csl` with `A_local[2*_A_pad]`/`B_local[2*_B_pad]`; add c=2 local GEMM loop | AC-2 | coding | task1 |
| task4 | Implement custom `@fmovh` two-tree broadcasts for A (x-axis) and B (y-axis) in `pe.csl`; add WSE-3 `@initialize_queue` comptime | AC-3 | coding | task3 |
| task5 | Rewrite `run.py`: `--P --M --K --N` interface, pre-load 2×tiles per PE, MEMCPY_16BIT, cycle-count reporting | AC-5 | coding | task2, task4 |
| task6 | Run end-to-end; debug with simprint if needed; confirm "Correctness check PASSED" | AC-4 | coding | task4, task5 |

---

## Claude-Codex Deliberation

### Agreements
- Keep P×P grid — no extra PEs needed for memory-based 2.5D
- c=2 pre-loaded A+B tiles per PE; first 2 SUMMA steps are local (zero communication)
- `@fmovh`/`@faddh` replaces `@fmovs` for all tile wavelet transfers
- `simd_mode = .{ .simd_max = true }` on all fabric DSDs for fp16 throughput
- WSE-3 requires `@initialize_queue` in comptime for all queue/color pairs
- run.py interface matches MeshGEMM (`--P --M --K --N --warmup --repeats`)
- `two_tree_allreduce_y` in `comm_pe.csl` is the reference for the broadcast phase pattern

### Resolved Disagreements
- **Extra processors vs extra memory**: Original plan used P*2×P grid (extra processors). User clarified 2.5D means extra local memory on the same P×P grid. Plan updated accordingly.
- **tile_config vs @fmovh broadcast**: Original plan preserved tile_config routing. User requires @fmovh-based custom broadcast replacing tile_config for broadcast steps.

### Convergence Status
- Final Status: `converged`

---

## Pending User Decisions

All decisions resolved.

---

## Implementation Notes

### Code Style Requirements
- Implementation code and comments must NOT contain plan-specific terminology such as "AC-", "Milestone", "Step", "Phase", or similar workflow markers
- Use descriptive, domain-appropriate names: `a_local_tiles`, `b_local_tiles`, `local_steps`, `broadcast_steps`, `two_tree_bcast_y`, etc.

---

--- Original Design Draft Start ---

# 2.5D GEMM Experiment on Cerebras WSE (Memory-Based, Extra-Memory Interpretation)

## Goal

Keep the same P×P PE grid. Each PE pre-loads c=2 A-tile and B-tile pairs from host into local memory (48KB). First c=2 SUMMA steps run locally (no broadcasts). Remaining P-c=2 steps use custom @fmovh two-tree broadcast from comm_pe.csl pattern. run.py matches MeshGEMM interface (--P --M --K --N --warmup --repeats).

--- Original Design Draft End ---
