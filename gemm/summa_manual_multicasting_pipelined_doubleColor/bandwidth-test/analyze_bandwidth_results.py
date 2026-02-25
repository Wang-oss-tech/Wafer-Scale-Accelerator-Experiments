#!/usr/bin/env python3
"""
Analyze bandwidth test results and fit performance model.

Usage:
    python analyze_bandwidth_results.py <results_dir>
    
Example:
    python analyze_bandwidth_results.py bandwidth_results_20260127_143000
"""

import sys
import os
import re
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from typing import List, Dict, Tuple


def parse_result_file(filepath: Path) -> Dict:
    """Parse a single bandwidth test result file."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Extract configuration from filename
    # Format: h2d_4x4_k196_ch1.txt or d2h_4x4_k196_ch1.txt
    filename = filepath.stem
    parts = filename.split('_')
    direction = parts[0]  # 'h2d' or 'd2h'
    
    grid_match = re.search(r'(\d+)x(\d+)', filename)
    k_match = re.search(r'k(\d+)', filename)
    ch_match = re.search(r'ch(\d+)', filename)
    
    width = int(grid_match.group(1)) if grid_match else None
    height = int(grid_match.group(2)) if grid_match else None
    k = int(k_match.group(1)) if k_match else None
    channels = int(ch_match.group(1)) if ch_match else None
    
    # Extract cycles_send from output
    cycles_match = re.search(r'cycles_send\s*=\s*(\d+)', content)
    cycles = int(cycles_match.group(1)) if cycles_match else None
    
    # Extract bandwidth if present
    bw_match = re.search(r'bandwidth\s*=\s*([\d.]+)\s*GB/s', content)
    bandwidth_gbs = float(bw_match.group(1)) if bw_match else None
    
    # Calculate total wavelets
    wavelets = width * height * k if (width and height and k) else None
    
    return {
        'direction': direction,
        'width': width,
        'height': height,
        'k': k,
        'channels': channels,
        'cycles': cycles,
        'bandwidth_gbs': bandwidth_gbs,
        'wavelets': wavelets,
        'n_pes': width * height if (width and height) else None,
    }


def load_all_results(results_dir: Path) -> List[Dict]:
    """Load all result files from directory."""
    results = []
    
    for filepath in results_dir.glob('*.txt'):
        try:
            result = parse_result_file(filepath)
            if result['cycles'] is not None:
                results.append(result)
                print(f"Loaded: {filepath.name} -> {result['cycles']} cycles")
        except Exception as e:
            print(f"Warning: Could not parse {filepath.name}: {e}")
    
    return results


def fit_linear_model(results: List[Dict], direction: str) -> Tuple[np.ndarray, Dict]:
    """
    Fit linear model: cycles = α × wavelets + β × (w + h) + γ
    
    Returns:
        coeffs: [α, β, γ]
        metrics: R², RMSE, etc.
    """
    # Filter by direction
    data = [r for r in results if r['direction'] == direction]
    
    if len(data) < 3:
        print(f"Warning: Only {len(data)} {direction} samples, need at least 3")
        return None, {}
    
    # Prepare data
    X = []
    y = []
    
    for r in data:
        wavelets = r['wavelets']
        perimeter = r['width'] + r['height']
        X.append([wavelets, perimeter, 1])  # Features: [wavelets, w+h, constant]
        y.append(r['cycles'])
    
    X = np.array(X)
    y = np.array(y)
    
    # Fit using least squares
    coeffs, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
    
    # Calculate metrics
    y_pred = X @ coeffs
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    rmse = np.sqrt(np.mean((y - y_pred) ** 2))
    mape = np.mean(np.abs((y - y_pred) / y)) * 100
    
    metrics = {
        'r2': r2,
        'rmse': rmse,
        'mape': mape,
        'n_samples': len(data),
    }
    
    return coeffs, metrics, data, y_pred


def fit_bandwidth_model(results: List[Dict], direction: str) -> Tuple[float, float]:
    """
    Fit bandwidth model: cycles = overhead + wavelets / bandwidth
    
    Returns:
        (bandwidth_words_per_cycle, overhead_cycles)
    """
    data = [r for r in results if r['direction'] == direction]
    
    # Prepare data
    wavelets = np.array([r['wavelets'] for r in data])
    cycles = np.array([r['cycles'] for r in data])
    
    # Fit: cycles = β + wavelets / α
    # Rearrange: cycles = β + wavelets × (1/α)
    def model(wavelets, bandwidth, overhead):
        return overhead + wavelets / bandwidth
    
    # Initial guess
    p0 = [0.5, 500]  # bandwidth, overhead
    
    try:
        params, _ = curve_fit(model, wavelets, cycles, p0=p0)
        bandwidth, overhead = params
        
        # Calculate fit quality
        y_pred = model(wavelets, bandwidth, overhead)
        r2 = 1 - np.sum((cycles - y_pred)**2) / np.sum((cycles - np.mean(cycles))**2)
        
        return bandwidth, overhead, r2
    except:
        print(f"Warning: Could not fit bandwidth model for {direction}")
        return None, None, None


def plot_results(results: List[Dict], coeffs_h2d, coeffs_d2h, output_dir: Path):
    """Generate plots for analysis."""
    
    # Split by direction
    h2d_data = [r for r in results if r['direction'] == 'h2d']
    d2h_data = [r for r in results if r['direction'] == 'd2h']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Cycles vs Wavelets (H2D)
    if h2d_data and coeffs_h2d is not None:
        ax = axes[0, 0]
        wavelets = [r['wavelets'] for r in h2d_data]
        cycles = [r['cycles'] for r in h2d_data]
        ax.scatter(wavelets, cycles, alpha=0.6, label='Measured')
        
        # Plot fit
        w_range = np.linspace(min(wavelets), max(wavelets), 100)
        perimeter_avg = np.mean([r['width'] + r['height'] for r in h2d_data])
        fit_cycles = coeffs_h2d[0] * w_range + coeffs_h2d[1] * perimeter_avg + coeffs_h2d[2]
        ax.plot(w_range, fit_cycles, 'r--', label='Fitted model')
        
        ax.set_xlabel('Wavelets (w × h × k)')
        ax.set_ylabel('Cycles')
        ax.set_title('H2D: Cycles vs Wavelets')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot 2: Cycles vs Wavelets (D2H)
    if d2h_data and coeffs_d2h is not None:
        ax = axes[0, 1]
        wavelets = [r['wavelets'] for r in d2h_data]
        cycles = [r['cycles'] for r in d2h_data]
        ax.scatter(wavelets, cycles, alpha=0.6, label='Measured', color='orange')
        
        # Plot fit
        w_range = np.linspace(min(wavelets), max(wavelets), 100)
        perimeter_avg = np.mean([r['width'] + r['height'] for r in d2h_data])
        fit_cycles = coeffs_d2h[0] * w_range + coeffs_d2h[1] * perimeter_avg + coeffs_d2h[2]
        ax.plot(w_range, fit_cycles, 'r--', label='Fitted model')
        
        ax.set_xlabel('Wavelets (w × h × k)')
        ax.set_ylabel('Cycles')
        ax.set_title('D2H: Cycles vs Wavelets')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot 3: Bandwidth comparison
    ax = axes[1, 0]
    if h2d_data:
        h2d_bw = [r['wavelets'] / r['cycles'] for r in h2d_data]
        h2d_sizes = [r['n_pes'] for r in h2d_data]
        ax.scatter(h2d_sizes, h2d_bw, alpha=0.6, label='H2D', marker='o')
    if d2h_data:
        d2h_bw = [r['wavelets'] / r['cycles'] for r in d2h_data]
        d2h_sizes = [r['n_pes'] for r in d2h_data]
        ax.scatter(d2h_sizes, d2h_bw, alpha=0.6, label='D2H', marker='s', color='orange')
    
    ax.set_xlabel('Number of PEs (w × h)')
    ax.set_ylabel('Bandwidth (words/cycle)')
    ax.set_title('Bandwidth vs Grid Size')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Prediction error
    ax = axes[1, 1]
    if h2d_data and coeffs_h2d is not None:
        h2d_pred = [coeffs_h2d[0] * r['wavelets'] + coeffs_h2d[1] * (r['width'] + r['height']) + coeffs_h2d[2] for r in h2d_data]
        h2d_meas = [r['cycles'] for r in h2d_data]
        h2d_err = [(p - m) / m * 100 for p, m in zip(h2d_pred, h2d_meas)]
        ax.scatter(range(len(h2d_err)), h2d_err, alpha=0.6, label='H2D', marker='o')
    
    if d2h_data and coeffs_d2h is not None:
        d2h_pred = [coeffs_d2h[0] * r['wavelets'] + coeffs_d2h[1] * (r['width'] + r['height']) + coeffs_d2h[2] for r in d2h_data]
        d2h_meas = [r['cycles'] for r in d2h_data]
        d2h_err = [(p - m) / m * 100 for p, m in zip(d2h_pred, d2h_meas)]
        ax.scatter(range(len(d2h_err)), d2h_err, alpha=0.6, label='D2H', marker='s', color='orange')
    
    ax.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Prediction Error (%)')
    ax.set_title('Model Prediction Error')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'bandwidth_analysis.png', dpi=150)
    print(f"\nPlot saved: {output_dir / 'bandwidth_analysis.png'}")


def generate_report(results: List[Dict], coeffs_h2d, metrics_h2d, coeffs_d2h, metrics_d2h, output_dir: Path):
    """Generate text report."""
    
    report_path = output_dir / 'bandwidth_model_report.txt'
    
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("BANDWIDTH TEST RESULTS AND MODEL FITTING\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Total samples: {len(results)}\n")
        f.write(f"  H2D samples: {len([r for r in results if r['direction'] == 'h2d'])}\n")
        f.write(f"  D2H samples: {len([r for r in results if r['direction'] == 'd2h'])}\n\n")
        
        # H2D Model
        f.write("=" * 80 + "\n")
        f.write("H2D MODEL\n")
        f.write("=" * 80 + "\n\n")
        
        if coeffs_h2d is not None:
            α_h2d, β_h2d, γ_h2d = coeffs_h2d
            f.write(f"Linear Model: cycles = α × wavelets + β × (w + h) + γ\n\n")
            f.write(f"  α (per wavelet):   {α_h2d:>12.6f} cycles/word\n")
            f.write(f"  β (per perimeter): {β_h2d:>12.2f} cycles/(w+h)\n")
            f.write(f"  γ (base overhead): {γ_h2d:>12.2f} cycles\n\n")
            
            f.write(f"Derived metrics:\n")
            f.write(f"  Bandwidth:         {1/α_h2d:>12.3f} words/cycle\n")
            f.write(f"  Overhead per PE:   {β_h2d:>12.2f} cycles\n\n")
            
            f.write(f"Fit quality:\n")
            f.write(f"  R²:                {metrics_h2d['r2']:>12.4f}\n")
            f.write(f"  RMSE:              {metrics_h2d['rmse']:>12.2f} cycles\n")
            f.write(f"  MAPE:              {metrics_h2d['mape']:>12.2f}%\n\n")
        
        # D2H Model
        f.write("=" * 80 + "\n")
        f.write("D2H MODEL\n")
        f.write("=" * 80 + "\n\n")
        
        if coeffs_d2h is not None:
            α_d2h, β_d2h, γ_d2h = coeffs_d2h
            f.write(f"Linear Model: cycles = α × wavelets + β × (w + h) + γ\n\n")
            f.write(f"  α (per wavelet):   {α_d2h:>12.6f} cycles/word\n")
            f.write(f"  β (per perimeter): {β_d2h:>12.2f} cycles/(w+h)\n")
            f.write(f"  γ (base overhead): {γ_d2h:>12.2f} cycles\n\n")
            
            f.write(f"Derived metrics:\n")
            f.write(f"  Bandwidth:         {1/α_d2h:>12.3f} words/cycle\n")
            f.write(f"  Overhead per PE:   {β_d2h:>12.2f} cycles\n\n")
            
            f.write(f"Fit quality:\n")
            f.write(f"  R²:                {metrics_d2h['r2']:>12.4f}\n")
            f.write(f"  RMSE:              {metrics_d2h['rmse']:>12.2f} cycles\n")
            f.write(f"  MAPE:              {metrics_d2h['mape']:>12.2f}%\n\n")
        
        # Comparison
        if coeffs_h2d is not None and coeffs_d2h is not None:
            f.write("=" * 80 + "\n")
            f.write("H2D vs D2H COMPARISON\n")
            f.write("=" * 80 + "\n\n")
            
            bw_ratio = (1/coeffs_h2d[0]) / (1/coeffs_d2h[0])
            f.write(f"  Bandwidth ratio (H2D/D2H): {bw_ratio:.2f}×\n")
            f.write(f"  D2H is {1/bw_ratio:.2f}× slower than H2D\n\n")
        
        # Usage example
        f.write("=" * 80 + "\n")
        f.write("USAGE IN predict_memcpy.py\n")
        f.write("=" * 80 + "\n\n")
        
        if coeffs_h2d is not None:
            f.write(f"def predict_h2d_cycles(w, h, k):\n")
            f.write(f"    wavelets = w * h * k\n")
            f.write(f"    perimeter = w + h\n")
            f.write(f"    cycles = {α_h2d:.6f} * wavelets + {β_h2d:.2f} * perimeter + {γ_h2d:.2f}\n")
            f.write(f"    return int(cycles)\n\n")
        
        if coeffs_d2h is not None:
            f.write(f"def predict_d2h_cycles(w, h, k):\n")
            f.write(f"    wavelets = w * h * k\n")
            f.write(f"    perimeter = w + h\n")
            f.write(f"    cycles = {α_d2h:.6f} * wavelets + {β_d2h:.2f} * perimeter + {γ_d2h:.2f}\n")
            f.write(f"    return int(cycles)\n\n")
    
    print(f"\nReport saved: {report_path}")
    
    # Also print to stdout
    with open(report_path, 'r') as f:
        print("\n" + f.read())


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_bandwidth_results.py <results_dir>")
        sys.exit(1)
    
    results_dir = Path(sys.argv[1])
    
    if not results_dir.exists():
        print(f"Error: Directory not found: {results_dir}")
        sys.exit(1)
    
    print("=" * 80)
    print("Bandwidth Results Analysis")
    print("=" * 80)
    print(f"Results directory: {results_dir}\n")
    
    # Load results
    results = load_all_results(results_dir)
    
    if len(results) == 0:
        print("Error: No valid results found!")
        sys.exit(1)
    
    print(f"\nLoaded {len(results)} valid results\n")
    
    # Fit models
    print("Fitting H2D model...")
    coeffs_h2d, metrics_h2d, data_h2d, pred_h2d = fit_linear_model(results, 'h2d')
    
    print("Fitting D2H model...")
    coeffs_d2h, metrics_d2h, data_d2h, pred_d2h = fit_linear_model(results, 'd2h')
    
    # Generate outputs
    print("\nGenerating plots...")
    plot_results(results, coeffs_h2d, coeffs_d2h, results_dir)
    
    print("\nGenerating report...")
    generate_report(results, coeffs_h2d, metrics_h2d, coeffs_d2h, metrics_d2h, results_dir)
    
    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
