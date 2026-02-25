#!/usr/bin/env python3
"""
Predict memcpy cycles for SUMMA GEMM based on configuration.
"""

def predict_h2d_cycles(P, Mt, Kt, Nt, BW_h2d=0.868, overhead=500):
    """
    Predict Host-to-Device memcpy cycles.
    
    Args:
        P: Grid size (P×P PEs)
        Mt, Kt, Nt: Tile dimensions
        BW_h2d: H2D bandwidth (words/cycle), default from WSE-2 measurements
        overhead: Setup overhead (cycles)
    
    Returns:
        Estimated H2D cycles
    """
    n_pes = P * P
    words_per_pe = Mt * Kt + Kt * Nt  # A_tile + B_tile
    total_words = n_pes * words_per_pe
    
    base_cycles = total_words / BW_h2d
    total_cycles = base_cycles + overhead
    
    return int(total_cycles)


def predict_d2h_cycles(P, Mt, Nt, BW_d2h=0.298, overhead=1000):
    """
    Predict Device-to-Host memcpy cycles.
    
    Args:
        P: Grid size (P×P PEs)
        Mt, Nt: Tile dimensions  
        BW_d2h: D2H bandwidth (words/cycle), default from WSE-2 measurements
        overhead: Gather + unblock overhead (cycles)
    
    Returns:
        Estimated D2H cycles
    """
    n_pes = P * P
    words_per_pe = Mt * Nt  # C_tile only
    total_words = n_pes * words_per_pe
    
    base_cycles = total_words / BW_d2h
    total_cycles = base_cycles + overhead
    
    return int(total_cycles)


def predict_compute_cycles(P, Mt, Kt, Nt, detailed=False):
    """
    Predict compute cycles for SUMMA GEMM using refined instruction-level model.
    
    Based on detailed instruction log analysis (@4_1_instr.log):
    - Setup: 120 cycles per iteration (DSD config, loop init)
    - FMACS: (1 + Mt) cycles per FMACS
    - Overhead: 43 cycles per FMACS (loops, arithmetic, stalls)
    
    Args:
        P: Grid size (number of SUMMA steps)
        Mt, Kt, Nt: Tile dimensions
        detailed: Return detailed breakdown if True
    
    Returns:
        Estimated compute cycles (or dict if detailed=True)
    """
    # Setup overhead per iteration
    setup_per_iter = 120  # cycles (DSD config, loop init, router config)
    
    # FMACS per iteration
    fmacs_per_iter = Kt * Nt
    
    # Pure FMACS execution
    cycles_per_fmacs = 1 + Mt  # 1 issue + Mt execution
    fmacs_cycles_per_iter = fmacs_per_iter * cycles_per_fmacs
    
    # Loop overhead per FMACS (from instruction log analysis)
    # - Scalar load: 8 cyc
    # - Index arithmetic: 12 cyc
    # - DSD increment: 5 cyc
    # - Loop control: 8 cyc
    # - Branch overhead: 3 cyc
    # - Pipeline stalls: 7 cyc
    # Total: 43 cycles per FMACS
    overhead_per_fmacs = 43
    overhead_cycles_per_iter = fmacs_per_iter * overhead_per_fmacs
    
    # Total per iteration
    cycles_per_iter = setup_per_iter + fmacs_cycles_per_iter + overhead_cycles_per_iter
    
    # Total for P iterations
    total_cycles = P * cycles_per_iter
    
    if detailed:
        return {
            'total_cycles': int(total_cycles),
            'setup_cycles': int(P * setup_per_iter),
            'fmacs_cycles': int(P * fmacs_cycles_per_iter),
            'overhead_cycles': int(P * overhead_cycles_per_iter),
            'cycles_per_iter': int(cycles_per_iter),
            'fmacs_count': P * fmacs_per_iter,
            'overhead_factor': total_cycles / (P * fmacs_cycles_per_iter),
        }
    
    return int(total_cycles)


def predict_total_execution(P, Mt, Kt, Nt, 
                           BW_h2d=0.868, BW_d2h=0.298):
    """
    Predict total execution time for SUMMA GEMM.
    
    Returns:
        Dictionary with breakdown of cycles
    """
    h2d = predict_h2d_cycles(P, Mt, Kt, Nt, BW_h2d)
    d2h = predict_d2h_cycles(P, Mt, Nt, BW_d2h)
    compute_detail = predict_compute_cycles(P, Mt, Kt, Nt, detailed=True)
    compute = compute_detail['total_cycles']
    
    # Broadcast cycles (overlapped with compute in pipelined version)
    broadcast_per_step = (Mt * Kt + Kt * Nt) / 0.512  # 0.512 from broadcast BW
    broadcast_total = int(P * broadcast_per_step)
    
    # For non-pipelined: sequential
    sequential_total = h2d + compute + d2h
    
    # For pipelined: broadcast overlaps with compute
    pipelined_total = h2d + compute + d2h  # Broadcast already overlapped
    
    return {
        'h2d_cycles': h2d,
        'd2h_cycles': d2h,
        'compute_cycles': compute,
        'compute_detail': compute_detail,
        'broadcast_cycles': broadcast_total,
        'sequential_total': sequential_total,
        'pipelined_total': pipelined_total,
        'config': {
            'P': P,
            'Mt': Mt,
            'Kt': Kt,
            'Nt': Nt,
            'n_pes': P * P,
        }
    }


def print_prediction(result):
    """Pretty print prediction results."""
    print("=" * 70)
    print("SUMMA GEMM Performance Prediction (Refined Model)")
    print("=" * 70)
    
    cfg = result['config']
    print(f"\nConfiguration:")
    print(f"  Grid:        {cfg['P']} × {cfg['P']} = {cfg['n_pes']} PEs")
    print(f"  Tile sizes:  Mt={cfg['Mt']}, Kt={cfg['Kt']}, Nt={cfg['Nt']}")
    
    print(f"\nCycle Breakdown:")
    print(f"  H2D memcpy:     {result['h2d_cycles']:>8,} cycles")
    print(f"  D2H memcpy:     {result['d2h_cycles']:>8,} cycles")
    print(f"  Compute:        {result['compute_cycles']:>8,} cycles")
    
    # Show detailed compute breakdown
    comp = result['compute_detail']
    print(f"    ├─ Setup:        {comp['setup_cycles']:>6,} cycles ({comp['setup_cycles']/comp['total_cycles']*100:>4.1f}%)")
    print(f"    ├─ Pure FMACS:   {comp['fmacs_cycles']:>6,} cycles ({comp['fmacs_cycles']/comp['total_cycles']*100:>4.1f}%)")
    print(f"    └─ Overhead:     {comp['overhead_cycles']:>6,} cycles ({comp['overhead_cycles']/comp['total_cycles']*100:>4.1f}%)")
    
    print(f"  Broadcast:      {result['broadcast_cycles']:>8,} cycles (overlapped)")
    
    print(f"\nTotal Execution:")
    print(f"  Sequential:     {result['sequential_total']:>8,} cycles")
    print(f"  Pipelined:      {result['pipelined_total']:>8,} cycles")
    
    speedup = result['sequential_total'] / result['pipelined_total']
    print(f"  Speedup:        {speedup:>8.2f}x")
    
    print("\nBreakdown by Phase:")
    total = result['pipelined_total']
    print(f"  H2D:     {result['h2d_cycles']/total*100:>6.2f}%")
    print(f"  Compute: {result['compute_cycles']/total*100:>6.2f}%")
    print(f"  D2H:     {result['d2h_cycles']/total*100:>6.2f}%")
    
    print(f"\nCompute Efficiency:")
    print(f"  Overhead factor: {comp['overhead_factor']:.2f}×")
    print(f"  Cycles/FMACS:    {comp['total_cycles']/comp['fmacs_count']:.1f}")


if __name__ == '__main__':
    import sys
    
    # Example: Your current configuration
    if len(sys.argv) == 1:
        print("Using default configuration (4, 14, 14, 14)")
        P, Mt, Kt, Nt = 4, 14, 14, 14
    elif len(sys.argv) == 5:
        P = int(sys.argv[1])
        Mt = int(sys.argv[2])
        Kt = int(sys.argv[3])
        Nt = int(sys.argv[4])
    else:
        print("Usage: python predict_memcpy.py [P Mt Kt Nt]")
        print("Example: python predict_memcpy.py 4 14 14 14")
        sys.exit(1)
    
    result = predict_total_execution(P, Mt, Kt, Nt)
    print_prediction(result)
    
    # Compare with measurements if default config
    if P == 4 and Mt == 14:
        print("\n" + "=" * 70)
        print("Comparison with Measurements (P=4, Mt=Kt=Nt=14)")
        print("=" * 70)
        measured = {
            'h2d': 7226,
            'd2h': 10522,
            'compute': 45783,
            'total': 64013
        }
        
        print(f"\n{'Phase':<15} {'Predicted':>12} {'Measured':>12} {'Error':>12}")
        print("-" * 70)
        print(f"{'H2D':<15} {result['h2d_cycles']:>12,} {measured['h2d']:>12,} {abs(result['h2d_cycles']-measured['h2d'])/measured['h2d']*100:>11.1f}%")
        print(f"{'D2H':<15} {result['d2h_cycles']:>12,} {measured['d2h']:>12,} {abs(result['d2h_cycles']-measured['d2h'])/measured['d2h']*100:>11.1f}%")
        print(f"{'Compute':<15} {result['compute_cycles']:>12,} {measured['compute']:>12,} {abs(result['compute_cycles']-measured['compute'])/measured['compute']*100:>11.1f}%")
        print(f"{'Total':<15} {result['pipelined_total']:>12,} {measured['total']:>12,} {abs(result['pipelined_total']-measured['total'])/measured['total']*100:>11.1f}%")
