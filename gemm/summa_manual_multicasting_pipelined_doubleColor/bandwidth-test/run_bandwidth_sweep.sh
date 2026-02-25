#!/usr/bin/env bash
#
# Bandwidth Sweep Experiment Suite
# 
# This script systematically measures H2D and D2H bandwidth across different configurations
# to fit an accurate memcpy performance model.
#
# Usage (on cer-usn-02):
#   ./run_bandwidth_sweep.sh

set -e

RESULTS_DIR="bandwidth_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

echo "========================================="
echo "Bandwidth Sweep Experiment Suite"
echo "========================================="
echo "Results directory: $RESULTS_DIR"
echo ""

# Compilation parameters (fixed)
FABRIC_DIMS="12,7"
FABRIC_OFFSETS="4,1"

# Function to compile for a given configuration
compile_config() {
    local width=$1
    local height=$2
    local pe_length=$3
    local channels=$4
    local out_dir="out_${width}x${height}_k${pe_length}_ch${channels}"
    
    echo "Compiling: ${width}×${height} grid, k=${pe_length}, channels=${channels}"
    
    cslc ./src/bw_sync_layout.csl \
        --arch wse2 \
        --fabric-dims=${FABRIC_DIMS} \
        --fabric-offsets=${FABRIC_OFFSETS} \
        --params=width:${width},height:${height},pe_length:${pe_length} \
        --params=C0_ID:0,C1_ID:1,C2_ID:2,C3_ID:3,C4_ID:4 \
        -o=${out_dir} \
        --memcpy \
        --channels=${channels} \
        --width-west-buf=0 \
        --width-east-buf=0
    
    echo "$out_dir"
}

# Function to run H2D test
run_h2d() {
    local width=$1
    local height=$2
    local pe_length=$3
    local channels=$4
    local loop_count=$5
    local out_dir=$6
    
    local result_file="${RESULTS_DIR}/h2d_${width}x${height}_k${pe_length}_ch${channels}.txt"
    
    echo "Running H2D: ${width}×${height}, k=${pe_length}, channels=${channels}, loops=${loop_count}"
    
    cs_python ./run.py \
        -m=${width} \
        -n=${height} \
        -k=${pe_length} \
        --latestlink ${out_dir} \
        --channels=${channels} \
        --width-west-buf=0 \
        --width-east-buf=0 \
        --run-only \
        --loop_count=${loop_count} \
        2>&1 | tee "$result_file"
    
    echo "Results saved to: $result_file"
}

# Function to run D2H test
run_d2h() {
    local width=$1
    local height=$2
    local pe_length=$3
    local channels=$4
    local loop_count=$5
    local out_dir=$6
    
    local result_file="${RESULTS_DIR}/d2h_${width}x${height}_k${pe_length}_ch${channels}.txt"
    
    echo "Running D2H: ${width}×${height}, k=${pe_length}, channels=${channels}, loops=${loop_count}"
    
    cs_python ./run.py \
        -m=${width} \
        -n=${height} \
        -k=${pe_length} \
        --latestlink ${out_dir} \
        --channels=${channels} \
        --width-west-buf=0 \
        --width-east-buf=0 \
        --run-only \
        --loop_count=${loop_count} \
        --d2h \
        2>&1 | tee "$result_file"
    
    echo "Results saved to: $result_file"
}

# Experiment configurations
# Format: "width height pe_length channels loop_count"
declare -a configs=(
    # Vary grid size (fixed k=196, channels=1)
    "2 2 196 1 10"
    "4 4 196 1 10"
    "6 6 196 1 10"
    "8 8 196 1 10"
    
    # Vary data per PE (fixed grid=4×4, channels=1)
    "4 4 98 1 10"
    "4 4 196 1 10"
    "4 4 392 1 10"
    "4 4 784 1 10"
    
    # Vary channels (fixed grid=4×4, k=196)
    "4 4 196 1 10"
    "4 4 196 2 10"
    "4 4 196 4 10"
    "4 4 196 8 10"
    
    # Non-square grids (fixed k=196, channels=1)
    "2 4 196 1 10"
    "4 2 196 1 10"
    "2 8 196 1 10"
    "8 2 196 1 10"
    
    # Large configurations
    "8 8 392 1 10"
    "8 8 392 4 10"
)

echo ""
echo "========================================="
echo "Starting Experiments"
echo "========================================="
echo "Total configurations: ${#configs[@]}"
echo ""

# Run experiments
for config in "${configs[@]}"; do
    read -r width height pe_length channels loop_count <<< "$config"
    
    echo ""
    echo "========================================="
    echo "Configuration: ${width}×${height}, k=${pe_length}, ch=${channels}"
    echo "========================================="
    
    # Compile (only if not already compiled)
    out_dir=$(compile_config $width $height $pe_length $channels)
    
    # Run H2D test
    run_h2d $width $height $pe_length $channels $loop_count $out_dir
    
    # Run D2H test
    run_d2h $width $height $pe_length $channels $loop_count $out_dir
    
    echo "Completed: ${width}×${height}, k=${pe_length}, ch=${channels}"
done

echo ""
echo "========================================="
echo "All Experiments Complete!"
echo "========================================="
echo "Results saved in: $RESULTS_DIR"
echo ""
echo "Next steps:"
echo "  1. Run: python analyze_bandwidth_results.py $RESULTS_DIR"
echo "  2. Review fitted model parameters"
echo "  3. Update predict_memcpy.py with new constants"
echo ""
