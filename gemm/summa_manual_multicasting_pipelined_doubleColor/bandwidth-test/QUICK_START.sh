#!/usr/bin/env bash
#
# Quick Start: Bandwidth Experiments
#
# This file contains the exact commands to run the bandwidth sweep
# and analyze results.
#

echo "========================================="
echo "Bandwidth Experiment Quick Start"
echo "========================================="
echo ""

# Step 1: SSH to hardware
echo "STEP 1: Connect to WSE-2 hardware"
echo "-----------------------------------------"
echo "Run this command:"
echo ""
echo "  ssh cer-usn-02"
echo ""
echo "Then navigate to bandwidth-test directory:"
echo ""
echo "  cd /home/william_wang_yale/R_2.6.0/csl-examples/experiments/gemm/summa_manual_multicasting_pipelined_doubleColor/bandwidth-test"
echo ""

# Step 2: Run sweep
echo ""
echo "STEP 2: Run bandwidth sweep"
echo "-----------------------------------------"
echo "This will test ~15 configurations (H2D + D2H each)"
echo "Expected runtime: 30-60 minutes"
echo ""
echo "Run this command:"
echo ""
echo "  ./run_bandwidth_sweep.sh"
echo ""
echo "Or to run in background:"
echo ""
echo "  nohup ./run_bandwidth_sweep.sh > sweep.log 2>&1 &"
echo "  tail -f sweep.log  # Monitor progress"
echo ""

# Step 3: Analyze
echo ""
echo "STEP 3: Analyze results"
echo "-----------------------------------------"
echo "After sweep completes, analyze the results:"
echo ""
echo "  python analyze_bandwidth_results.py bandwidth_results_<timestamp>"
echo ""
echo "Example:"
echo ""
echo "  python analyze_bandwidth_results.py bandwidth_results_20260127_143000"
echo ""

# Step 4: Update model
echo ""
echo "STEP 4: Update prediction model"
echo "-----------------------------------------"
echo "Copy the fitted parameters from bandwidth_model_report.txt"
echo "into ../predict_memcpy.py"
echo ""
echo "The report will show code like:"
echo ""
echo "  def predict_h2d_cycles(w, h, k):"
echo "      wavelets = w * h * k"
echo "      perimeter = w + h"
echo "      cycles = 1.152 * wavelets + 45.3 * perimeter + 387.1"
echo "      return int(cycles)"
echo ""

# Alternative: Test single configuration
echo ""
echo "========================================="
echo "ALTERNATIVE: Quick Test (Single Config)"
echo "========================================="
echo ""
echo "To test a single configuration before full sweep:"
echo ""
echo "  cd /path/to/bandwidth-test"
echo "  ./commands_wse2.sh  # Compile default config"
echo "  cs_python run.py -m=4 -n=4 -k=196 --loop_count=10  # H2D"
echo "  cs_python run.py -m=4 -n=4 -k=196 --loop_count=10 --d2h  # D2H"
echo ""

echo ""
echo "========================================="
echo "For full documentation, see:"
echo "  EXPERIMENT_GUIDE.md"
echo "========================================="
echo ""
