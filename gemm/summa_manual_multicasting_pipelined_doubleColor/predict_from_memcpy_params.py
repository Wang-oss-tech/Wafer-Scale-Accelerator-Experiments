#!/usr/bin/env python3
"""
Predict memcpy cycles directly from runner.memcpy_h2d/d2h parameters.
"""

def predict_h2d_from_params(x_start, y_start, width, height, count_per_pe,
                            BW_h2d=0.868, overhead=500):
    """
    Predict H2D memcpy cycles from runner.memcpy_h2d parameters.
    
    Args:
        x_start, y_start: Starting PE coordinates (usually 0, 0)
        width, height: PE grid dimensions (w, h)
        count_per_pe: Number of elements per PE (Mt*Kt or Kt*Nt)
        BW_h2d: Bandwidth in words/cycle
        overhead: Setup overhead in cycles
    
    Returns:
        Predicted cycles
    
    Example:
        # From: runner.memcpy_h2d(sym_A, data, 0, 0, w, h, Mt*Kt, ...)
        cycles = predict_h2d_from_params(0, 0, w, h, Mt*Kt)
    """
    n_pes = width * height
    total_words = n_pes * count_per_pe
    
    base_cycles = total_words / BW_h2d
    total_cycles = base_cycles + overhead
    
    return int(total_cycles)


def predict_d2h_from_params(x_start, y_start, width, height, count_per_pe,
                            BW_d2h=0.298, overhead=1000):
    """
    Predict D2H memcpy cycles from runner.memcpy_d2h parameters.
    
    Args:
        x_start, y_start: Starting PE coordinates (usually 0, 0)
        width, height: PE grid dimensions (w, h)
        count_per_pe: Number of elements per PE (Mt*Nt)
        BW_d2h: Bandwidth in words/cycle
        overhead: Setup overhead in cycles
    
    Returns:
        Predicted cycles
    
    Example:
        # From: runner.memcpy_d2h(data, sym_C, 0, 0, w, h, Mt*Nt, ...)
        cycles = predict_d2h_from_params(0, 0, w, h, Mt*Nt)
    """
    n_pes = width * height
    total_words = n_pes * count_per_pe
    
    base_cycles = total_words / BW_d2h
    total_cycles = base_cycles + overhead
    
    return int(total_cycles)


def predict_from_run_py(w, h, Mt, Kt, Nt):
    """
    Predict all memcpy cycles from run.py configuration.
    
    This matches your exact run.py structure:
        runner.memcpy_h2d(sym_A, A3.ravel(), 0, 0, w, h, Mt*Kt, ...)
        runner.memcpy_h2d(sym_B, B3.ravel(), 0, 0, w, h, Kt*Nt, ...)
        runner.memcpy_d2h(C3_1d_u32, sym_C, 0, 0, w, h, Mt*Nt, ...)
    
    Args:
        w, h: Grid dimensions (typically P, P)
        Mt, Kt, Nt: Tile dimensions
    
    Returns:
        Dictionary with predictions
    """
    # H2D for A
    h2d_A = predict_h2d_from_params(0, 0, w, h, Mt*Kt)
    
    # H2D for B
    h2d_B = predict_h2d_from_params(0, 0, w, h, Kt*Nt)
    
    # Total H2D (A and B are transferred sequentially or overlapped)
    # Since both use nonblock=True, they may overlap, but conservatively add them
    h2d_total = h2d_A + h2d_B
    
    # D2H for C
    d2h_C = predict_d2h_from_params(0, 0, w, h, Mt*Nt)
    
    return {
        'h2d_A_cycles': h2d_A,
        'h2d_B_cycles': h2d_B,
        'h2d_total_cycles': h2d_total,
        'd2h_C_cycles': d2h_C,
        'config': {
            'grid': f"{w}×{h}",
            'n_pes': w * h,
            'Mt': Mt,
            'Kt': Kt,
            'Nt': Nt,
            'A_per_pe': Mt * Kt,
            'B_per_pe': Kt * Nt,
            'C_per_pe': Mt * Nt,
        }
    }


def analyze_memcpy_call(call_string):
    """
    Parse and predict from a memcpy call string.
    
    Example:
        call = "runner.memcpy_h2d(sym_A, A3.ravel(), 0, 0, 4, 4, 196, ...)"
        analyze_memcpy_call(call)
    """
    import re
    
    # Extract parameters: x, y, w, h, count
    pattern = r'memcpy_([hd]2[hd])\([^,]+,[^,]+,\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+)'
    match = re.search(pattern, call_string)
    
    if not match:
        return None
    
    direction = match.group(1)
    x_start = int(match.group(2))
    y_start = int(match.group(3))
    width = int(match.group(4))
    height = int(match.group(5))
    count = int(match.group(6))
    
    if direction == 'h2d':
        cycles = predict_h2d_from_params(x_start, y_start, width, height, count)
        direction_name = "Host → Device"
    else:
        cycles = predict_d2h_from_params(x_start, y_start, width, height, count)
        direction_name = "Device → Host"
    
    return {
        'direction': direction_name,
        'grid': f"{width}×{height}",
        'n_pes': width * height,
        'count_per_pe': count,
        'total_words': width * height * count,
        'predicted_cycles': cycles
    }


if __name__ == '__main__':
    import sys
    
    print("=" * 70)
    print("Memcpy Prediction from run.py Parameters")
    print("=" * 70)
    
    # Your configuration from run.py
    w = 4   # width
    h = 4   # height
    Mt = 14
    Kt = 14
    Nt = 14
    
    if len(sys.argv) == 6:
        w = int(sys.argv[1])
        h = int(sys.argv[2])
        Mt = int(sys.argv[3])
        Kt = int(sys.argv[4])
        Nt = int(sys.argv[5])
    
    result = predict_from_run_py(w, h, Mt, Kt, Nt)
    
    print(f"\nConfiguration:")
    print(f"  Grid:        {result['config']['grid']} = {result['config']['n_pes']} PEs")
    print(f"  Tile sizes:  Mt={Mt}, Kt={Kt}, Nt={Nt}")
    
    print(f"\nData per PE:")
    print(f"  A_tile:      {result['config']['A_per_pe']} elements (Mt×Kt)")
    print(f"  B_tile:      {result['config']['B_per_pe']} elements (Kt×Nt)")
    print(f"  C_tile:      {result['config']['C_per_pe']} elements (Mt×Nt)")
    
    print(f"\nMemcpy Predictions:")
    print(f"  H2D (A):     {result['h2d_A_cycles']:>8,} cycles")
    print(f"  H2D (B):     {result['h2d_B_cycles']:>8,} cycles")
    print(f"  H2D (Total): {result['h2d_total_cycles']:>8,} cycles")
    print(f"  D2H (C):     {result['d2h_C_cycles']:>8,} cycles")
    
    # Show the exact memcpy calls that would be made
    print(f"\nEquivalent to these run.py calls:")
    print(f"  runner.memcpy_h2d(sym_A, ..., 0, 0, {w}, {h}, {Mt*Kt}, ...)")
    print(f"  runner.memcpy_h2d(sym_B, ..., 0, 0, {w}, {h}, {Kt*Nt}, ...)")
    print(f"  runner.memcpy_d2h(..., sym_C, 0, 0, {w}, {h}, {Mt*Nt}, ...)")
    
    # Compare with measurements if default config
    if w == 4 and h == 4 and Mt == 14:
        print("\n" + "=" * 70)
        print("Comparison with Measurements")
        print("=" * 70)
        
        measured_h2d = 7226
        measured_d2h = 10522
        
        print(f"\n{'Phase':<15} {'Predicted':>12} {'Measured':>12} {'Error':>12}")
        print("-" * 70)
        
        # Note: measured H2D is for A+B combined
        error_h2d = abs(result['h2d_total_cycles'] - measured_h2d) / measured_h2d * 100
        error_d2h = abs(result['d2h_C_cycles'] - measured_d2h) / measured_d2h * 100
        
        print(f"{'H2D (A+B)':<15} {result['h2d_total_cycles']:>12,} {measured_h2d:>12,} {error_h2d:>11.1f}%")
        print(f"{'D2H (C)':<15} {result['d2h_C_cycles']:>12,} {measured_d2h:>12,} {error_d2h:>11.1f}%")
    
    print("\n" + "=" * 70)
    print("Usage Examples:")
    print("=" * 70)
    print("\n# Predict from your run.py configuration:")
    print("python3 predict_from_memcpy_params.py 4 4 14 14 14")
    print("\n# Or import and use in Python:")
    print("from predict_from_memcpy_params import predict_from_run_py")
    print("result = predict_from_run_py(w=4, h=4, Mt=14, Kt=14, Nt=14)")
    print("print(f'H2D: {result[\"h2d_total_cycles\"]} cycles')")
    print("print(f'D2H: {result[\"d2h_C_cycles\"]} cycles')")
