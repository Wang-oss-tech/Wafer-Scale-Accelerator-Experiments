#!/usr/bin/env bash
# Runs all fp16 GEMV experiment configurations and appends output to results.txt

set -eo pipefail

OUT="results.txt"
> "$OUT"
mkdir -p compile_out

run_config() {
    local P=$1 M=$2 N=$3

    echo "========================================" | tee -a "$OUT"
    echo "Grid: ${P} x ${P} PEs" | tee -a "$OUT"
    echo "Tile sizes: Mt=$((M / P)), Nt=$((N / P))" | tee -a "$OUT"
    echo "Full matrix: x(1x${N}) * A(${M}x${N}) = y(1x${M})" | tee -a "$OUT"
    echo "========================================" | tee -a "$OUT"

    python3 compile.py "$P" "$M" "$N" false 2>&1 | tee -a "$OUT"
    python3 run_hw.py --P "$P" --M "$M" --N "$N" 2>&1 | tee -a "$OUT"

    echo "" | tee -a "$OUT"
}

# fp16_gemv stores x_src[M], b_src[M], final_result[M] on every PE.
# Per-PE data cost = M*8 bytes; 48 KB limit leaves ~33 KB for data + code.
# Use Mt=Nt=6 per PE: M = 6*P for each P.
# M: 720, 1440, 2160, 2880, 3600 → per-PE data = 5760–28800 bytes (well within 48 KB)

# Mt=Nt=6 per PE
run_config 120  720  720
run_config 240 1440 1440
run_config 360 2160 2160
run_config 480 2880 2880
run_config 600 3600 3600

# Mt=Nt=12 per PE (2x larger tiles) — safe only for P<=240 (M*8 <= 23 KB)
run_config 120 1440 1440
run_config 240 2880 2880

echo "All experiments complete. Results saved to $OUT"
