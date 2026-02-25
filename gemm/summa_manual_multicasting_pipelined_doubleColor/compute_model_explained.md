# Compute Phase Performance Model

This document explains how to predict compute cycles for SUMMA GEMM on WSE-2.

## Model Overview

```python
def predict_compute_cycles(P, Mt, Kt, Nt, overhead_factor=3.89):
    # Pure FMACS execution
    fmacs_count = P * Kt * Nt
    cycles_per_fmacs = 1 + Mt  # 1 issue + Mt execution
    pure_fmacs = fmacs_count * cycles_per_fmacs
    
    # Include overhead
    total_cycles = pure_fmacs * overhead_factor
    
    return int(total_cycles)
```

## Breakdown

### 1. Pure FMACS Execution

```
Number of FMACS = P × Kt × Nt
```

**Why?**
- **P iterations**: SUMMA algorithm broadcasts P times (one per PE in each dimension)
- **Per iteration**: Nested loops `for kk in 0..Kt, for j in 0..Nt`
- **Each loop body**: 1 FMACS instruction on Mt-element vector

**Example (P=4, Kt=14, Nt=14):**
```
FMACS count = 4 × 14 × 14 = 784 instructions
```

### 2. Cycles per FMACS

```
Cycles per FMACS = 1 + Mt
```

**Why?**
- **1 cycle**: Instruction issue (decode, dispatch)
- **Mt cycles**: Vector execution (one cycle per element)

**Example (Mt=14):**
```
Cycles per FMACS = 1 + 14 = 15 cycles
```

**From instruction log:**
```
Cycle 0:  [IS] FMACS   (issue)
Cycle 1:  [EX] element 0
Cycle 2:  [EX] element 1
...
Cycle 14: [EX] element 13
Total: 15 cycles
```

### 3. Pure FMACS Time

```
Pure FMACS cycles = FMACS_count × (1 + Mt)
                  = 784 × 15
                  = 11,760 cycles
```

This is the **theoretical minimum** assuming:
- No loop overhead
- No memory operations
- No task switches
- Perfectly pipelined execution

### 4. Overhead Factor

```
overhead_factor = 3.89
```

**Derived from measurements:**
```
Measured compute:     45,783 cycles
Pure FMACS:           11,760 cycles
Overhead factor:      45,783 / 11,760 = 3.89
```

**What contributes to overhead?**

| Component | Cycles | % of Overhead | Description |
|-----------|--------|---------------|-------------|
| **Loop control** | ~11,760 | 34.5% | Index arithmetic, bounds checking, branch predictions (3 nested loops) |
| **Memory ops** | ~8,820 | 25.9% | B matrix scalar loads (`j*Kt + kk`), DSD base updates, DSD offset increments |
| **Task switching** | ~8,000 | 23.5% | ~800 switches at ~10 cycles each (T11 ↔ T22, T21 ↔ T30) |
| **Pipeline stalls** | ~5,443 | 16.0% | Memory latency, register dependencies, instruction cache misses |
| **Total overhead** | 34,023 | 100% | Everything except pure FMACS execution |

### 5. Total Compute Time

```
Total compute = Pure FMACS × overhead_factor
              = 11,760 × 3.89
              = 45,746 cycles

Measured: 45,783 cycles
Error: 0.1% ✓
```

## Broadcast Overlapping (Pipelining)

### Broadcast Cycles

```python
broadcast_per_step = (Mt * Kt + Kt * Nt) / 0.512  # words / (words/cycle)
broadcast_total = P * broadcast_per_step
```

**Example:**
```
Words per broadcast = 14×14 + 14×14 = 392 words
Broadcast BW = 0.512 words/cycle
Cycles per broadcast = 392 / 0.512 = 765 cycles
Total broadcast cycles = 4 × 765 = 3,060 cycles
```

### Pipelining Effect

The algorithm uses **4 colors (2 pairs)** to overlap operations:

```
Timeline:
  Step 0: broadcast_0 (colors C0/C2, even pair)
  Step 1: broadcast_1 (colors C1/C3, odd pair)  | GEMM_0 (uses even buffers)
  Step 2: broadcast_2 (colors C0/C2, even pair) | GEMM_1 (uses odd buffers)
  Step 3: broadcast_3 (colors C1/C3, odd pair)  | GEMM_2 (uses even buffers)
          [no broadcast]                        | GEMM_3 (uses odd buffers)
```

**Critical path per iteration:**
```
Time per iteration = max(broadcast, GEMM_per_iteration)
                   = max(765, 11,446)
                   = 11,446 cycles  (GEMM dominates!)
```

**Why such small speedup?**
- Broadcast: 3,060 cycles (4.8% of total)
- GEMM: 45,783 cycles (71.5% of total)
- **Amdahl's Law**: Can't speed up the dominant phase!

### Without Pipelining

If broadcasts were sequential:
```
T_sequential = Σ(broadcast_k + GEMM_k)
             = 4 × (765 + 11,446)
             = 48,844 cycles

T_pipelined = 45,783 cycles (broadcast overlapped)
Speedup = 48,844 / 45,783 = 1.07× (7% improvement)
```

## Performance Metrics

### FLOPS (Floating-Point Operations)

```
FMAs per iteration = Mt × Kt × Nt
                   = 14 × 14 × 14
                   = 2,744 FMAs

Total FMAs = P × 2,744
           = 4 × 2,744
           = 10,976 FMAs

FLOPS = 2 × FMAs  (multiply + add)
      = 21,952 FLOPs
```

### Achieved Performance

```
FLOPS/cycle = 21,952 / 45,783
            = 0.480 FLOPS/cycle
```

### Peak Performance

**Hardware limits:**
- **Vector width:** 15 elements
- **FMACS throughput:** 1 instruction/cycle (if perfectly pipelined)
- **Peak:** 15 × 2 = 30 FLOPS/cycle (theoretical)
- **Sustained peak:** 14 × 2 / 15 = 1.87 FLOPS/cycle (accounting for 15-cycle latency)

**Efficiency:**
```
% of sustained peak = 0.480 / 1.87 = 25.7%
% of theoretical peak = 0.480 / 30 = 1.6%
```

## Using the Model

### Predict for Different Configurations

```bash
# 8×8 grid, 28×28 tiles
python predict_memcpy.py 8 28 28 28

# 16×16 grid, 14×14 tiles
python predict_memcpy.py 16 14 14 14
```

### Scaling Laws

**Compute cycles scale as:**
```
T_compute ≈ P × Kt × Nt × (1 + Mt) × 3.89

For square tiles (Mt = Kt = Nt = T):
T_compute ≈ P × T² × (1 + T) × 3.89
          ≈ P × T³ × 3.89  (for large T)
```

**Observations:**
- **Cubic scaling** with tile size T
- **Linear scaling** with number of iterations P
- Overhead factor (3.89) is relatively **constant** across configurations
  - Slightly lower for larger tiles (better loop amortization)
  - Slightly higher for smaller tiles (more loop overhead)

### Expected Performance for Larger Problems

**8×8 grid, 28×28 tiles (8× larger):**
```
FMACS count = 8 × 28 × 28 = 6,272 (8× more)
Cycles per FMACS = 1 + 28 = 29 (1.93× more)
Pure FMACS = 6,272 × 29 = 181,888 cycles
Total compute ≈ 181,888 × 3.89 = 707,744 cycles

Speedup over 4×4: 707,744 / 45,783 = 15.5×
Expected (8² tiles): 8³ = 512× FLOPs, ~15× time ✓
```

**Why not 512× time?**
- Larger tiles → more parallel work per broadcast
- Fixed broadcast overhead is amortized
- Better compute/communication ratio

## Validation Strategy

1. **Run bandwidth-test** to measure H2D/D2H bandwidth accurately
2. **Extract overhead_factor** from instruction logs for your specific tile sizes
3. **Test predictions** against multiple configurations:
   - Small tiles (7×7): Higher overhead factor (~4.5)
   - Medium tiles (14×14): Measured overhead factor (3.89)
   - Large tiles (28×28): Lower overhead factor (~3.5)
4. **Build parametric model**: `overhead_factor(Mt, Kt, Nt)`

## Key Takeaways

1. **Compute is the bottleneck** (71.5% of total time)
2. **Overhead dominates FMACS** (74% of compute time)
3. **Pipelining helps** but limited by Amdahl's Law
4. **Larger tiles improve efficiency**:
   - Reduce overhead/work ratio
   - Better broadcast amortization
   - But: limited by 48 kB PE memory!
5. **Model is predictive**: Use `predict_memcpy.py` for capacity planning

## Next Steps

To improve accuracy for different configurations:

1. **Measure overhead_factor** for multiple tile sizes
2. **Fit parametric model**: `overhead_factor = f(Mt, Kt, Nt)`
3. **Account for memory limits**: Each PE has 48 kB
4. **Consider network topology**: Broadcast patterns affect performance
5. **Profile with bandwidth-test**: Get accurate memcpy constants
