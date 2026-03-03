# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This directory contains research experiments implementing GEMM (matrix multiplication) algorithms on the Cerebras Wafer Scale Engine (WSE). The primary focus is variants of the SUMMA algorithm on a P×P PE grid, exploring different broadcast strategies and pipelining.

## Build and Run

Every experiment has `commands_wse2.sh` and/or `commands_wse3.sh`. Run them directly:

```bash
cd <experiment_dir>
./commands_wse2.sh   # compile + simulate
```

Manual compilation:
```bash
cslc --arch=wse2 ./layout.csl \
  --fabric-dims=<P+7>,<P+2> --fabric-offsets=4,1 \
  --params=P:<p>,Mt:<m>,Kt:<k>,Nt:<n> \
  --memcpy --channels=1 -o out
cs_python run.py --name out
```

Fabric dims rule for simulation: `--fabric-dims=<P+7>,<P+2> --fabric-offsets=4,1` (e.g., P=4 → `11,6`). WSE-3 at P=180 uses `187,184`.

Enable instruction tracing (slows simulation significantly):
```bash
export SINGULARITYENV_SIMFABRIC_DEBUG=inst_trace
```

For WaferLLM sub-experiments, use the module's `run_sim.sh` or `run_device.sh` scripts.

## Directory Structure

| Directory | Description |
|-----------|-------------|
| `gemm-collectives_2d/` | SUMMA using SDK `<collectives_2d>` library — the canonical clean implementation |
| `summa_manual_multicasting/` | SUMMA with runtime routing via `<tile_config>` color config registers |
| `summa_manual_multicasting_pipelined/` | Pipelined SUMMA: broadcast(k+1) overlaps with GEMM(k) using double buffers |
| `summa_manual_multicasting_pipelined_doubleColor/` | Pipelined with dual-color broadcasts; includes performance model docs and analysis scripts |
| `summa_overlapping_experiment/` | Overlapping SUMMA variant using collectives_2d |
| `broadcast_paper/` | Standalone 2D broadcast primitives explored for a paper |
| `fabin_reference/` | Reference SUMMA implementation (baseline) |
| `previous_manual/` | Older manual implementations (`summa_manual_original/`, `summa_manual_pipelined/`) |
| `topic-03-streaming-wavelet-data/` | CSL streaming wavelet data example (unrelated to GEMM) |
| `useful_python_scripts/` | Shared simulation trace parsers |
| `WaferLLM/` | WaferLLM OSDI'25 paper code: MeshGEMM, MeshGEMV, Prefill, Decode, Shift, Resize |

## SUMMA Algorithm Architecture

All GEMM experiments implement variants of SUMMA on a P×P PE grid:

- **Tile layout**: Each PE holds `A_tile[Mt*Kt]`, `B_tile[Kt*Nt]`, `C_tile[Mt*Nt]` (column-major f32)
- **Full matrix**: `M = Mt*P`, `K = Kt*P`, `N = Nt*P`
- **P steps**: At step `k`, column-`k` PEs broadcast A tiles along their row; row-`k` PEs broadcast B tiles along their column; all PEs compute `C_tile += A * B`
- **Local GEMM**: Uses `@fmacs` DSD instruction — inner loop over columns of B, outer loop over K

### Task/Color Coordination Pattern

```
x_done() task  → @activate(compute_task_id)
y_done() task  → @unblock(compute_task_id)
compute() task → does GEMM, then calls main() for next step or @activate(EXIT)
```
`compute_task_id` starts blocked; requires both broadcasts to complete (activate + unblock = runs once).

## Two Broadcast Approaches

### 1. `collectives_2d` Library (gemm-collectives_2d, summa_overlapping_experiment)
Uses `<collectives_2d/params>` in layout and `<collectives_2d/pe>` in PE code. Colors 0,1 for x-axis; colors 4,5 for y-axis. Cleaner but less flexible.

### 2. Manual Tile Config (summa_manual_multicasting*)
Dynamically reconfigures color routing registers each SUMMA step using `<tile_config>`:

```csl
const tile_config = @import_module("<tile_config>");
const addr = @as(u16, addresses.COLOR_CONFIG) + @get_int(c) * word_size;
// Write input/output bits to configure direction per step
```

Color config register bits: `INPUT_RAMP=0x200`, `INPUT_N=0x100`, `INPUT_S=0x080`, `INPUT_E=0x040`, `INPUT_W=0x020`; `OUTPUT_RAMP=0x10`, `OUTPUT_N=0x08`, `OUTPUT_S=0x04`, `OUTPUT_E=0x02`, `OUTPUT_W=0x01`. Use `MASK_CLEAR_IO=0xfc00` to preserve control bits.

## Matrix Distribution for Memcpy (Cliff Distribution)

Host matrices must be distributed across PEs in column-major local tiles:

```python
# A is M×K, distribute to h×w PE grid with Mt×Kt tiles (column-major)
A1 = A.reshape(h, Mt, w, Kt)   # (h, lh, w, lw)
A2 = A1.transpose(0, 2, 3, 1)  # (h, w, lw, lh) → col-major local
A3 = A2.reshape(h, w, Mt*Kt)   # h-w-l flat
runner.memcpy_h2d(sym_A, A3.ravel(), 0, 0, w, h, Mt*Kt, ...)

# Inverse for C retrieval
C3 = C3_1d.reshape(h, w, Nt, Mt)
C2 = C3.transpose(0, 3, 1, 2)  # (h, Mt, w, Nt)
C1 = C2.reshape(M, N)
```

## Performance Instrumentation

48-bit cycle counter via `<time>` module (3 u16 words = `tsc_size_words`):

```csl
timestamp.enable_tsc();
timestamp.get_timestamp(&tscStartBuffer);
// ... work ...
timestamp.get_timestamp(&tscEndBuffer);
timestamp.disable_tsc();
```

Timestamps must be packed into f32 bit patterns for memcpy transfer (only f32 DMA is supported). Packing layout: `f32[0]={start[1],start[0]}`, `f32[1]={end[0],start[2]}`, `f32[2]={end[2],end[1]}`.

**Clock skew correction**: subtract per-PE reference timestamp plus Manhattan distance offset (`px + py`) before computing elapsed cycles.

**Empirical timings** (P=4, Mt=Kt=Nt=14, WSE-2): broadcast ~600 cycles, GEMM ~11,000 cycles, H2D memcpy ~7,226 cycles, D2H memcpy ~10,522 cycles.

## Analysis Scripts

Located in `useful_python_scripts/` and `summa_manual_multicasting_pipelined_doubleColor/`:

- `parse_color_link_timeline.py` — parses simulation wavelet/color link trace logs
- `parse_task_timeline.py` — parses task execution timeline from sim traces
- `predict_memcpy.py` / `predict_from_memcpy_params.py` — memcpy bandwidth prediction models

Trace logs generated when `SINGULARITYENV_SIMFABRIC_DEBUG=inst_trace` is set. Simulation output goes to `simfab_traces/` and `sim.log`.

## WaferLLM Sub-Project

OSDI'25 paper implementation. Each module (`MeshGEMM`, `MeshGEMV`, `Prefill`, `Decode`, `Shift`, `Resize`) has `WSE-2/` and `WSE-3/` subdirectories with `run_sim.sh` / `run_wse*.sh` scripts. Uses a custom `comm_lib/` (`comm_layout.csl` + `comm_pe.csl`) instead of the SDK collectives. MeshGEMM uses f16 tiles with even-padding to keep DSDs in-bounds when tile size is odd.
