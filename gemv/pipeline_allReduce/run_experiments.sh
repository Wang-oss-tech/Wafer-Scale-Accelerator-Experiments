#!/usr/bin/env bash
# Runs all pipeline-allreduce GEMV experiment configurations and appends output to results.txt

set -eo pipefail

OUT="results.txt"
> "$OUT"
mkdir -p compile_out

WARMUP="${1:-10}"
REPEATS="${2:-100}"

run_config() {
    local P=$1 M=$2 N=$3

    echo "========================================" | tee -a "$OUT"
    echo "Grid: ${P} x ${P} PEs" | tee -a "$OUT"
    echo "Tile sizes: Mt=$((M / P)), Nt=$((N / P))" | tee -a "$OUT"
    echo "Full matrix: x(1x${M}) * W(${M}x${N}) = y(1x${N})" | tee -a "$OUT"
    echo "warmup=${WARMUP}, repeats=${REPEATS}" | tee -a "$OUT"
    echo "========================================" | tee -a "$OUT"

    python3 compile.py "$P" "$M" "$N" false 2>&1 | tee -a "$OUT"
    python3 run_hw.py --P "$P" --M "$M" --N "$N" --warmup "$WARMUP" --repeats "$REPEATS" 2>&1 | tee -a "$OUT"

    echo "" | tee -a "$OUT"
}

# Same square sweeps as waferllm-gemv; run as:
#   ./run_experiments.sh            # warmup=10, repeats=100
#   ./run_experiments.sh 0 1        # one-shot verification run

# GEMV ~4K  (M=N=4320 for P=120,240,360,480; 4800 for P=600)
run_config 120 4320 4320
run_config 240 4320 4320
run_config 360 4320 4320
run_config 480 4320 4320
run_config 600 4800 4800

# GEMV ~8K  (M=N=8640 for P=120,240,360,480; 7200 for P=600)
run_config 120 8640 8640
run_config 240 8640 8640
run_config 360 8640 8640
run_config 480 8640 8640
run_config 600 7200 7200

# GEMV 16K  (M=N=14400, divisible by all P)
run_config 120 14400 14400
run_config 240 14400 14400
run_config 360 14400 14400
run_config 480 14400 14400
run_config 600 14400 14400

echo "All experiments complete. Results saved to $OUT"
