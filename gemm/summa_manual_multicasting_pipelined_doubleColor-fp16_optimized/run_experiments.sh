#!/usr/bin/env bash
# Runs all experiment configurations and appends output to results.txt

set -e

OUT="results.txt"
> "$OUT"  # clear/create file

run_config() {
    local P=$1 Mt=$2 Kt=$3 Nt=$4
    local M=$((P * Mt))
    local K=$((P * Kt))
    local N=$((P * Nt))

    echo "========================================" | tee -a "$OUT"
    echo "Grid: ${P} x ${P} PEs" | tee -a "$OUT"
    echo "Tile sizes: Mt=${Mt}, Kt=${Kt}, Nt=${Nt}" | tee -a "$OUT"
    echo "Full matrices: A(${M}x${K}) @ B(${K}x${N}) = C(${M}x${N})" | tee -a "$OUT"
    echo "========================================" | tee -a "$OUT"

    python3 compile.py "$P" "$Mt" "$Kt" "$Nt" false 2>&1 | tee -a "$OUT"
    python3 appliance_launch.py --P "$P" --M "$M" --K "$K" --N "$N" --warmup 0 --repeats 1 2>&1 | tee -a "$OUT"

    echo "" | tee -a "$OUT"
}

# M=K=N=2160
run_config 180 12 12 12
run_config 360  6  6  6
run_config 540  4  4  4
run_config 720  3  3  3

# M=K=N=4320
run_config 360 12 12 12
run_config 540  8  8  8
run_config 720  6  6  6

# M=K=N=8640
run_config 360 24 24 24
run_config 540 16 16 16
run_config 720 12 12 12

echo "All experiments complete. Results saved to $OUT"
