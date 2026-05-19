"""Performance model for 2.5D GEMM and MeshGEMM, side-by-side.

Reuses the calibrated compute/comm helpers from
`gemm/summa_manual_multicasting_pipelined_doubleColor-fp16_optimized/performance_model_meshgemm.py`.

2.5D total cycles = per-layer Cannon + cross-layer reduction:

    Cannon_25D = startup_shift_cycles(q, Mt, Kt_layer)
               + q × (compute_iter(Mt, Kt_layer, Nt) + comm_exposed_iter(Mt, Kt_layer, Nt))

    Reduction = 2q × per_substep            # q substeps for x-axis + q for y-axis
    per_substep = TILE_CFG + DSD_SETUP + TASK_OVERHEAD
                + P_phys × HOP_LATENCY      # broadcast wavelet across the row
                + 2 × Mt × Nt               # broadcast DMA + accumulate at partner

MeshGEMM total cycles (same problem at the same P_phys):

    MeshGEMM = startup_shift_cycles(P, Mt_mg, Kt_mg)
             + P × (compute_iter(Mt_mg, Kt_mg, Nt_mg) + comm_exposed_iter(Mt_mg, Kt_mg, Nt_mg))

Where for the same (M, K, N, P_phys):
    - MeshGEMM:  Mt_mg = M/P_phys,  Kt_mg = K/P_phys,    Nt_mg = N/P_phys
    - 2.5D:      Mt    = M/q,        Kt_layer = K/(c·q), Nt    = N/q
                 (q = P_phys/sqrt_c, c = sqrt_c²)

Run:
    python3 performance_model_25d.py --P_phys 480 --sqrt_c 2 --M 4096 --K 4096 --N 4096
    python3 performance_model_25d.py --sweep    # try several configs
"""

import argparse
import math
import os
import sys

# Pull MeshGEMM helpers from the doubleColor experiment dir.
HERE = os.path.dirname(os.path.abspath(__file__))
MESHGEMM_MODEL_DIR = os.path.normpath(os.path.join(
    HERE, "..", "summa_manual_multicasting_pipelined_doubleColor-fp16_optimized"))
sys.path.insert(0, MESHGEMM_MODEL_DIR)

from performance_model_meshgemm import (  # noqa: E402
    compute_iter,
    comm_exposed_iter,
    startup_shift_cycles,
    x_elems,
    w_elems,
    fmach_issue_period,
)


# ============================================================================
# 2.5D-specific tunable constants — calibrate against measured cycles.
# ============================================================================

# Per-substep overhead for the broadcast-based cross-layer reduction.
RED_TILE_CONFIG_COST = 30.0     # color_config.reset_routes call
RED_DSD_SETUP_COST   = 20.0     # async DSD issue setup
RED_TASK_OVERHEAD    = 30.0     # task fire + activate
RED_HOPS_PER_CYCLE   = 1.0      # broadcast wavelet propagation rate (cycles per fabric hop)
RED_FADD_BYTES_PER_CYCLE = 1.0  # @faddh accumulation rate at the partner PE


# ============================================================================
# 2.5D model
# ============================================================================

def reduction_per_substep_cycles(P_phys, Mt, Nt):
    """Cost of one broadcast substep in our Phase 2 reduction.

    Source PE broadcasts Mt*Nt elements via fabout. Receiver PEs receive via
    fabin. Fabric latency is dominated by the wavelet propagation across the
    full P_phys-PE row before the partner can start consuming. The partner PE
    then accumulates with @faddh. We treat the mask cost (a few cycles) as
    negligible and absorbed into TASK_OVERHEAD.
    """
    overhead = RED_TILE_CONFIG_COST + RED_DSD_SETUP_COST + RED_TASK_OVERHEAD
    fabric_latency = P_phys * RED_HOPS_PER_CYCLE
    broadcast_dma  = Mt * Nt
    accumulate     = Mt * Nt * RED_FADD_BYTES_PER_CYCLE
    return overhead + fabric_latency + broadcast_dma + accumulate


def reduction_cycles(P_phys, sqrt_c, Mt, Nt):
    """Total reduction cost: q substeps for x-axis + q for y-axis."""
    q = P_phys // sqrt_c
    return 2 * q * reduction_per_substep_cycles(P_phys, Mt, Nt)


def cannon_25d_cycles(P_phys, sqrt_c, Mt, Kt_layer, Nt):
    """Per-layer Cannon GEMM (same shape as MeshGEMM, on a q×q sub-mesh)."""
    q = P_phys // sqrt_c
    return startup_shift_cycles(q, Mt, Kt_layer) + q * (
        compute_iter(Mt, Kt_layer, Nt)
        + comm_exposed_iter(Mt, Kt_layer, Nt)
    )


def kernel_25d_cycles(P_phys, sqrt_c, M, K, N):
    """Total 2.5D = per-layer Cannon + cross-layer reduction."""
    q = P_phys // sqrt_c
    c = sqrt_c ** 2

    if M % q != 0 or N % q != 0 or K % (c * q) != 0:
        raise ValueError(
            f"Bad dims: M={M} N={N} K={K} need M%q==0, N%q==0, K%(c*q)==0 "
            f"(q={q}, c={c}, c*q={c*q})"
        )

    Mt = M // q
    Nt = N // q
    Kt_layer = K // (c * q)

    cannon = cannon_25d_cycles(P_phys, sqrt_c, Mt, Kt_layer, Nt)
    reduction = reduction_cycles(P_phys, sqrt_c, Mt, Nt)
    return cannon, reduction, cannon + reduction


# ============================================================================
# MeshGEMM wrapper (operates at full P_phys mesh).
# ============================================================================

def kernel_meshgemm_cycles(P_phys, M, K, N):
    if M % P_phys != 0 or K % P_phys != 0 or N % P_phys != 0:
        raise ValueError(f"Bad dims: M={M} N={N} K={K} need divisible by P_phys={P_phys}")

    Mt = M // P_phys
    Kt = K // P_phys
    Nt = N // P_phys

    startup = startup_shift_cycles(P_phys, Mt, Kt)
    cannon  = P_phys * (compute_iter(Mt, Kt, Nt) + comm_exposed_iter(Mt, Kt, Nt))
    return startup + cannon


# ============================================================================
# Comparison harness
# ============================================================================

def predict_and_compare(P_phys, sqrt_c, M, K, N):
    q = P_phys // sqrt_c
    c = sqrt_c ** 2

    mg = kernel_meshgemm_cycles(P_phys, M, K, N)
    cannon_25d, reduction_25d, total_25d = kernel_25d_cycles(P_phys, sqrt_c, M, K, N)

    print(f"  P_phys={P_phys}  sqrt_c={sqrt_c} (c={c}, q={q})")
    print(f"  M={M}, K={K}, N={N}")
    print(f"  ┌─────────────┬──────────────┬────────────┬────────────┐")
    print(f"  │ Kernel      │ Cannon (cyc) │ Red (cyc)  │ Total      │")
    print(f"  ├─────────────┼──────────────┼────────────┼────────────┤")
    print(f"  │ MeshGEMM    │ {mg:>12,} │ {0:>10,} │ {mg:>10,} │")
    print(f"  │ 2.5D        │ {cannon_25d:>12,.0f} │ {reduction_25d:>10,.0f} │ {total_25d:>10,.0f} │")
    print(f"  └─────────────┴──────────────┴────────────┴────────────┘")
    print(f"  2.5D speedup vs MeshGEMM: {mg / total_25d:.2f}×")
    print(f"  Reduction as % of 2.5D total: {100 * reduction_25d / total_25d:.1f}%")
    if mg > total_25d:
        print(f"  → 2.5D wins by {(mg - total_25d) / mg * 100:.1f}% of MeshGEMM time")
    else:
        print(f"  → MeshGEMM wins by {(total_25d - mg) / total_25d * 100:.1f}% of 2.5D time")
    print()
    return mg, total_25d


def sweep():
    """Sweep configurations to find when 2.5D wins."""
    cases = [
        # (P_phys, sqrt_c, M, K, N, label)
        (8,    2, 16,    16,    16,   "small smoke test (matches our P=8 simulator run)"),
        (16,   2, 32,    128,   32,   "P=16 simulator run baseline"),
        (60,   2, 240,   240,   240,  "moderate-scale prefill"),
        (180,  2, 360,   360,   360,  "moderate-scale, square"),
        (240,  2, 480,   480,   480,  "approaching production"),
        (480,  2, 960,   960,   960,  "production-scale, modest dims"),
        (480,  2, 4800,  4800,  4800, "production-scale, paper-style dims"),
        (480,  4, 4800,  4800,  4800, "production-scale, sqrt_c=4 (deeper K split)"),
        (660,  2, 4620,  4620,  4620, "WaferLLM 8B-prefill scale"),
    ]
    print("=" * 80)
    print("2.5D vs MeshGEMM performance-model sweep")
    print("=" * 80)
    print()
    for P_phys, sqrt_c, M, K, N, label in cases:
        print(f"### {label}")
        try:
            predict_and_compare(P_phys, sqrt_c, M, K, N)
        except ValueError as e:
            print(f"  SKIPPED: {e}")
            print()


# ============================================================================
# CLI
# ============================================================================

def main():
    p = argparse.ArgumentParser(description="2.5D vs MeshGEMM performance model")
    p.add_argument("--P_phys", type=int, default=0)
    p.add_argument("--sqrt_c", type=int, default=2)
    p.add_argument("--M", type=int, default=0)
    p.add_argument("--K", type=int, default=0)
    p.add_argument("--N", type=int, default=0)
    p.add_argument("--sweep", action="store_true",
                   help="Sweep predefined configs instead of single point")
    args = p.parse_args()

    if args.sweep or args.P_phys == 0:
        sweep()
        return

    predict_and_compare(args.P_phys, args.sqrt_c, args.M, args.K, args.N)


if __name__ == "__main__":
    main()
