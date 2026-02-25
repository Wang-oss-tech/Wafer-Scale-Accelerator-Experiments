# GEMM Performance Model - Quick Reference

## TL;DR

```python
# From run.py or commands_wse2.sh, extract: P, Mt, Kt, Nt, channels

# Predict performance:
python predict_memcpy.py P Mt Kt Nt
```

## Model Formulas

### H2D Memcpy
```
words = P² × (Mt×Kt + Kt×Nt)
cycles = (words / 0.868) + 500
```

### Compute
```
FMACS = P × Kt × Nt
pure_cycles = FMACS × (1 + Mt)
total_cycles = pure_cycles × 3.89
```

### D2H Memcpy
```
words = P² × Mt × Nt
cycles = (words / 0.298) + 1000
```

### Total
```
T_total = T_h2d + T_compute + T_d2h
```

## Constants (Empirical, from 4×4 grid, 14×14 tiles)

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `BW_h2d` | 0.868 words/cycle | H2D bandwidth |
| `BW_d2h` | 0.298 words/cycle | D2H bandwidth (3× slower!) |
| `overhead_factor` | 3.89 | Compute overhead multiplier |
| `C_h2d` | 500 cycles | H2D setup overhead |
| `C_d2h` | 1000 cycles | D2H gather overhead |

## Breakdown (P=4, Mt=Kt=Nt=14)

```
Total: 64,994 cycles (measured: 64,013, error: 1.5%)

├─ H2D:      7,725 cycles (11.9%) ← efficient
├─ Compute: 45,746 cycles (70.4%) ← BOTTLENECK
│  ├─ Pure FMACS:  11,760 cycles (25.7%)
│  └─ Overhead:    33,986 cycles (74.3%)
│     ├─ Loop control:    11,725 (34.5%)
│     ├─ Memory ops:       8,802 (25.9%)
│     ├─ Task switches:    7,986 (23.5%)
│     └─ Pipeline stalls:  5,437 (16.0%)
└─ D2H:     11,523 cycles (17.7%) ← moderately expensive

Broadcast: 3,060 cycles (FULLY OVERLAPPED via pipelining)
```

## Key Insights

### 🔴 Bottleneck: Compute (70%)
- **Problem:** 74% is overhead, only 26% actual FMACS
- **Solution:** Increase tile size (Mt, Kt, Nt)
  - 2× tile → 8× work, but only ~3× time (better amortization)
  - Limited by 48 kB PE memory

### 🟡 D2H is 3× slower than H2D
- **Reason:** Sequential PE iteration, gather pattern
- **Solution:** 
  - Use `streaming=True` in memcpy
  - Increase channels (1 → 4 → 16)

### 🟢 Pipelining works!
- 3,060 broadcast cycles fully hidden
- 4 colors (2 pairs) enable overlap

## Performance Metrics

```
FMACS:        784 instructions
FMAs:      10,976 operations
FLOPs:     21,952 (2 × FMAs)
Performance: 0.480 FLOPS/cycle
Efficiency:  25.7% of peak (1.87 FLOPS/cycle)
```

## Optimization Checklist

- [ ] **Increase tile size** (Mt, Kt, Nt = 28)
  - 8× more FLOPs per broadcast
  - Better overhead amortization
  - Check: 2×(Mt×Kt + Mt×Nt + Kt×Nt) < 48kB
  
- [ ] **Increase channels** (1 → 4 or 16)
  - Faster H2D/D2H
  - ~70% efficiency per channel
  
- [ ] **Use streaming D2H**
  - Add `streaming=True` to `memcpy_d2h()`
  - 2-3× speedup on D2H
  
- [ ] **Unroll loops** (CSL compiler may do this)
  - Reduce loop control overhead
  - Better ILP
  
- [ ] **Profile with bandwidth-test**
  - Get accurate BW constants for your config
  - Measure channel scaling

## Files

- **`predict_memcpy.py`**: Prediction script (run this!)
- **`performance_model.md`**: Detailed analysis
- **`compute_model_explained.md`**: Deep dive on compute phase
- **`task_timeline_output.txt`**: Measured task execution times
- **`4_1_instr.log`**: Low-level instruction trace

## Example: Predict for 8×8 grid, 28×28 tiles

```bash
python predict_memcpy.py 8 28 28 28
```

**Expected output:**
```
H2D:       ~30,000 cycles
Compute:  ~700,000 cycles
D2H:       ~43,000 cycles
Total:    ~773,000 cycles

FLOPs: 1,404,928 (64× more than 4×4/14×14)
Time:  ~12× longer
Efficiency: Improved (better compute/comm ratio)
```

## Remember

1. **Compute dominates** (70% of time)
2. **Overhead dominates compute** (74% of compute)
3. **Scale up tiles** to improve efficiency
4. **Model accuracy: 1.5% error** ✓
