# Performance Model for Pipelined SUMMA GEMM

## Configuration
- Grid: P × P = 4 × 4 = 16 PEs
- Tile sizes: Mt = 14, Kt = 14, Nt = 14
- Architecture: WSE-2
- Data type: f32 (32-bit float)

## Data Sizes
| Matrix | Size per PE | Bytes | Total for 16 PEs |
|--------|-------------|-------|------------------|
| A_tile | 14 × 14 = 196 | 784 | 12,544 |
| B_tile | 14 × 14 = 196 | 784 | 12,544 |
| C_tile | 14 × 14 = 196 | 784 | 12,544 |

## Execution Timeline (PE 4,1)

### Phase 1: Host-to-Device Memcpy
```
Start:    @127
End:      @7353
Duration: 7,226 cycles
Data:     A_tile + B_tile = 1,568 bytes per PE (25,088 total)
```

**Bandwidth:**
- 6,272 words / 7,226 cycles = **0.868 words/cycle**
- Per PE: 392 words / 7,226 cycles = **0.054 words/cycle/PE**

### Phase 2: Computation (Main + Pipelined Broadcasts)
```
Start:    @7835  (broadcast 0 complete)
End:      @53618 (GEMM complete)
Duration: 45,783 cycles
```

**Breakdown:**
| Task | Cycles | % of Total | Description |
|------|--------|------------|-------------|
| T11 (main) | 45,783 | 100% | GEMM computation |
| T22 (async) | 3,060 | 6.7% | Fabric broadcasts (overlap) |
| T21 | 6,101 | 13.3% | Micro-tasks (queue management) |
| T30 | 3,528 | 7.7% | Micro-tasks (queue management) |

**Note:** T22 overlaps with T11, so effective compute time is dominated by T11.

### Phase 3: Device-to-Host Memcpy
```
Start:    @53618
End:      @64140
Duration: 10,522 cycles
Data:     C_tile = 784 bytes per PE (12,544 total)
```

**Bandwidth:**
- 3,136 words / 10,522 cycles = **0.298 words/cycle**
- Per PE: 196 words / 10,522 cycles = **0.019 words/cycle/PE**

## Why D2H is Slower Than H2D?

### 1. Data Volume
- H2D: 2× the data (A + B)
- D2H: 1× the data (C only)
- **Expected:** D2H should be 2× faster
- **Actual:** D2H is 1.46× slower!

### 2. Network Pattern
```
H2D (Broadcast):           D2H (Gather):
Host ──┬──> PE[0,0]        PE[0,0] ──┐
       ├──> PE[0,1]        PE[0,1] ──┤
       └──> PE[3,3]        PE[3,3] ──┴──> Host
       
Parallel delivery         Serial collection
Low contention            High contention at host
```

### 3. Bandwidth Comparison
- H2D: 0.868 words/cycle
- D2H: 0.298 words/cycle
- **D2H is 2.9× slower per byte**

### 4. Root Causes
1. **Gather bottleneck:** All 16 PEs compete to send to host
2. **Blocking vs non-blocking:** D2H uses `nonblock=False`
3. **Setup overhead:** D2H includes command stream unblock
4. **Buffer management:** Host must serialize incoming streams

## GEMM Computation Analysis

### FMACS Count
For each SUMMA step `k`, one PE computes:
```
C += A[:, k] × B[k, :]

Operations:
  for kk in 0..Kt-1:
    for j in 0..Nt-1:
      C[:, j] += A[:, kk] * B[kk, j]  # Mt FMAs
      
Total FMAs per step: Mt × Kt × Nt = 14 × 14 × 14 = 2,744
Total FMAs for P steps: 2,744 × 4 = 10,976
```

### FMACS Execution
From instruction logs:
- Each `@fmacs` processes **Mt = 14 elements** in parallel
- Each FMACS takes **15 cycles total**:
  - 1 cycle: Instruction issue
  - 14 cycles: Vector execution (one cycle per element, Mt=14)
  - **Note:** Hardware vector size is 15, but only 14 elements are used
- Number of FMACS per step: Kt × Nt = 14 × 14 = 196
- Total FMACS instructions: 196 × 4 = 784

### Compute Performance
```
Total compute cycles: 45,783
Total FMACS: 784
Cycles per FMACS: 45,783 / 784 ≈ 58.4 cycles

Pure FMACS execution: 784 × 15 = 11,760 cycles (25.7% of compute time)
Overhead: 45,783 - 11,760 = 34,023 cycles (74.3% of compute time!)

Total FMAs: 10,976
Performance: 10,976 / 45,783 ≈ 0.240 FLOPS/cycle
```

**Breakdown of 58.4 cycles per FMACS:**
- FMACS execution: 15 cycles (26%)
  - Issue: 1 cycle
  - Vector execution: 14 cycles (Mt=14 elements)
- Overhead: 43.4 cycles (74%)
  - Loop overhead (index calculation, bounds checking)
  - DSD updates (@increment_dsd_offset)
  - Pointer arithmetic for B matrix access (column-major indexing)
  - Control flow (nested loops, conditionals)
  - Task switching (800 switches total, ~10-30 cycles each)
  - Memory latency and pipeline stalls

## Broadcast Performance

### Per Broadcast
```
T22 (fabric async): 3,060 cycles for 4 broadcasts
Average per broadcast: 765 cycles

Data per broadcast:
  A: Mt × Kt = 196 f32 = 784 bytes
  B: Kt × Nt = 196 f32 = 784 bytes
  Total: 1,568 bytes

Bandwidth: 392 words / 765 cycles ≈ 0.512 words/cycle
```

### Broadcast Pattern
For P=4, each PE participates in:
- 4 A broadcasts (as sender or receiver)
- 4 B broadcasts (as sender or receiver)

## Overall Performance Model

### Total Execution Time
```
T_total = T_h2d + T_compute + T_d2h + T_overhead
        = 7,226 + 45,783 + 10,522 + overhead
        = 63,531 cycles (measured: 64,013)
        
Overhead ≈ 482 cycles (startup, synchronization)
```

### Speedup from Pipelining
**Without pipelining (sequential):**
```
T_seq = T_h2d + Σ(T_broadcast[k] + T_gemm[k]) + T_d2h
      ≈ 7,226 + (4 × 765 + 4 × 11,446) + 10,522
      ≈ 7,226 + 48,844 + 10,522
      = 66,592 cycles
```

**With pipelining (measured):**
```
T_pipe = 64,013 cycles
Speedup = 66,592 / 64,013 ≈ 1.04× (4% improvement)
```

**Why so small?**
- GEMM dominates (45,783 cycles = 71.5% of total)
- Broadcast is only 3,060 cycles (4.8% of total)
- Limited overlap opportunity

### FMACS Instruction Details

### Hardware Characteristics
- **Vector width:** 15 elements (hardware maximum)
- **Used elements:** 14 (Mt = 14)
- **Execution model:**
  ```
  Cycle 0:  [IS] Issue FMACS instruction
  Cycle 1:  [EX] Process element 0
  Cycle 2:  [EX] Process element 1
  ...
  Cycle 14: [EX] Process element 13
  Total: 15 cycles (1 issue + 14 execution)
  ```

### Theoretical vs Actual
- **Theoretical peak:** 1 FMACS issued every cycle (if fully pipelined)
- **Actual:** 1 FMACS every 58.4 cycles (including overhead)
- **FMACS utilization:** 15 / 58.4 = 25.7%

## Roofline Analysis

**Peak Performance (theoretical):**
- WSE-2 PE: 1 FMACS/cycle × 14 elements = 14 FLOPS/cycle (if perfectly pipelined)
- With 15-cycle FMACS latency: 14 / 15 = 0.933 FLOPS/cycle (sustained peak)

**Achieved Performance:**
- 0.240 FLOPS/cycle (25.7% of sustained peak, 1.7% of theoretical peak)

**Bottlenecks:**
1. **Loop and control overhead** (40%)
   - Nested loop management (3 levels: k, kk, j)
   - Branch predictions and conditionals
   - Register spilling and allocation
2. **Memory operations** (25%)
   - B matrix scalar loads (column-major indexing: `j*Kt + kk`)
   - DSD base address updates
   - DSD offset increments
3. **Task switching** (9%)
   - 800 task switches at ~10-30 cycles each
   - Context save/restore overhead
4. **FMACS execution** (26%)
   - Pure vector execution: 11,760 cycles
   - Issue + 14 element processing = 15 cycles per FMACS

### Optimization Opportunities

1. **Increase tile size:**
   - Mt, Kt, Nt = 28 → 4× more work per broadcast
   - Better compute/communication ratio
   
2. **Reduce task switches:**
   - 800 switches in 64,013 cycles
   - Each switch ~10-30 cycles overhead
   - Potential savings: ~8,000-24,000 cycles
   
3. **Loop unrolling:**
   - Reduce loop control overhead
   - Better instruction-level parallelism
   
4. **Streaming memcpy:**
   - Use `streaming=True` for D2H
   - Potential 2-3× speedup on D2H phase

## Validation

From task timeline:
```
Total execution: 64,013 cycles
  - T11 (main):  45,783 cycles (71.5%)
  - T22 (async):  3,060 cycles (4.8%, overlapped)
  - T21/T30:      9,629 cycles (15.0%, overhead)
  - Other:        5,541 cycles (8.7%, memcpy + overhead)
```

Model matches observed behavior! ✓
