# Bandwidth Experiment Guide

This guide walks you through running systematic bandwidth experiments to fit an accurate memcpy performance model.

## Overview

We'll measure H2D and D2H bandwidth across different configurations and fit a model:

```
cycles = α × wavelets + β × (w + h) + γ

Where:
  α = cycles per wavelet (inverse bandwidth)
  β = overhead per PE (perimeter effect)
  γ = base overhead (setup, teardown)
```

## Prerequisites

- Access to `cer-usn-02` (WSE-2 hardware)
- Compiled CSL SDK
- Python 3 with numpy, scipy, matplotlib

## Step 1: Connect to Hardware

```bash
ssh cer-usn-02
cd /home/william_wang_yale/R_2.6.0/csl-examples/experiments/gemm/summa_manual_multicasting_pipelined_doubleColor/bandwidth-test
```

## Step 2: Run Bandwidth Sweep

The sweep script will:
- Test different grid sizes (2×2, 4×4, 6×6, 8×8)
- Test different data sizes per PE (98, 196, 392, 784 words)
- Test different channel counts (1, 2, 4, 8)
- Test non-square grids (2×4, 4×2, 2×8, 8×2)
- Run both H2D and D2H for each configuration
- Save results with timestamps

```bash
# Make script executable
chmod +x run_bandwidth_sweep.sh

# Run the sweep (takes ~30-60 minutes)
./run_bandwidth_sweep.sh
```

### What Gets Measured

For each configuration, the benchmark measures:
- `cycles_send`: Total cycles from first PE starts receiving to last PE finishes
- `bandwidth`: Computed bandwidth in GB/s
- Timing breakdown per PE (via TSC counters)

### Expected Output

```
Results saved in: bandwidth_results_20260127_143000/
  ├─ h2d_2x2_k196_ch1.txt
  ├─ d2h_2x2_k196_ch1.txt
  ├─ h2d_4x4_k196_ch1.txt
  ├─ d2h_4x4_k196_ch1.txt
  └─ ... (30+ files)
```

## Step 3: Analyze Results

Transfer results to analysis machine (if needed):

```bash
# From cer-usn-02
scp -r bandwidth_results_20260127_143000 <your-machine>:~/
```

Run analysis:

```bash
# Install dependencies if needed
pip install numpy scipy matplotlib

# Analyze results
python analyze_bandwidth_results.py bandwidth_results_20260127_143000
```

### Analysis Outputs

1. **`bandwidth_model_report.txt`**: Fitted model parameters
   ```
   H2D MODEL:
     α (per wavelet):    1.152467 cycles/word
     β (per perimeter):  45.32 cycles/(w+h)
     γ (base overhead):  387.12 cycles
     
     Bandwidth:          0.868 words/cycle
     R²:                 0.9973
     MAPE:               3.2%
   ```

2. **`bandwidth_analysis.png`**: Visualization plots
   - Cycles vs wavelets (H2D and D2H)
   - Bandwidth vs grid size
   - Model prediction errors

3. **Console output**: Python code ready to copy into `predict_memcpy.py`

## Step 4: Update Prediction Model

Copy the fitted functions into your performance model:

```python
# In predict_memcpy.py

def predict_h2d_cycles(w, h, k):
    """Predict H2D cycles using fitted model."""
    wavelets = w * h * k
    perimeter = w + h
    cycles = 1.152467 * wavelets + 45.32 * perimeter + 387.12
    return int(cycles)

def predict_d2h_cycles(w, h, k):
    """Predict D2H cycles using fitted model."""
    wavelets = w * h * k
    perimeter = w + h
    cycles = 3.354821 * wavelets + 67.89 * perimeter + 892.45
    return int(cycles)
```

## Step 5: Validate

Test the updated model:

```bash
python predict_memcpy.py 4 14 14 14
```

Compare predicted vs measured cycles from your GEMM runs.

## Troubleshooting

### Compilation Fails

If you get fabric dimension errors:
```bash
# Check available fabric dims
cs_python -c "from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime; print('OK')"

# Adjust FABRIC_DIMS in run_bandwidth_sweep.sh
FABRIC_DIMS="12,7"  # WSE-2
FABRIC_OFFSETS="4,1"
```

### Run Fails

If cs_python fails:
```bash
# Check hardware connection
cs_python -c "from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime; print('Connected')"

# Verify cmaddr if using --cmaddr flag
# Check run.py for required arguments
```

### Missing Dependencies

```bash
# On cer-usn-02
module load python3
pip install --user numpy scipy matplotlib
```

## Understanding the Model

### Why This Form?

```
cycles = α × wavelets + β × (w + h) + γ
```

- **α × wavelets**: Data transfer time (linear in data volume)
- **β × (w + h)**: Overhead per PE (perimeter of grid)
  - Reflects sequential iteration through PEs
  - Larger for D2H (gather) than H2D (broadcast)
- **γ**: Fixed overhead (setup, sync, teardown)

### Expected Parameter Ranges (WSE-2)

From previous measurements:

| Parameter | H2D | D2H | Ratio |
|-----------|-----|-----|-------|
| α (cyc/word) | 1.0-1.2 | 3.0-3.5 | ~3× |
| β (cyc/PE) | 30-50 | 50-100 | ~2× |
| γ (cyc) | 300-500 | 800-1200 | ~2.5× |
| Bandwidth (words/cyc) | 0.8-1.0 | 0.28-0.35 | ~3× |

D2H is consistently 2-3× slower due to gather pattern.

### Interpreting R² and MAPE

- **R² > 0.99**: Excellent fit, model explains 99%+ of variance
- **R² > 0.95**: Good fit
- **R² < 0.90**: Poor fit, consider different model form

- **MAPE < 5%**: Excellent accuracy
- **MAPE < 10%**: Good accuracy
- **MAPE > 15%**: Poor accuracy, model needs refinement

## Advanced: Channel Scaling

To model channel effects separately:

```python
# Fit per-channel models
def predict_h2d_with_channels(w, h, k, channels=1):
    base_cycles = α * w * h * k + β * (w + h) + γ
    # Channel scaling factor (empirical)
    scale = 1.0 / (1.0 + 0.7 * (channels - 1))
    return int(base_cycles * scale)
```

Analyze channel scaling from your sweep data to refine this.

## Next Steps

1. **Validate model** on held-out test cases
2. **Test edge cases**: Very small grids (1×1), very large grids (16×16)
3. **Test different data patterns**: Broadcast vs point-to-point
4. **Compare WSE-2 vs WSE-3** if available
5. **Document assumptions** and limitations

## References

- Bandwidth-test README: `../../benchmarks/bandwidth-test/README.rst`
- SDK docs: Cerebras SDK documentation
- Your GEMM analysis: `../REFINED_COMPUTE_MODEL.md`
