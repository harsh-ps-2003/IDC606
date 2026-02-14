#!/usr/bin/env python
"""
Run the full rigorous spectral entropy analysis on GraphCast weights.

This script:
1. Downloads the GraphCast checkpoint (~147 MB for 0.25deg model)
2. Runs the rigorous sensitivity analysis
3. Computes spectral entropy and power law fit
4. Generates visualizations
5. Outputs a summary table with distance labels

Usage:
    uv run python run_analysis.py
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add the project root to path
sys.path.insert(0, str(Path(__file__).parent))

from spectral_entropy import (
    # Mesh and geometry
    MESH_LEVELS,
    GRAPHCAST_CONFIGS,
    get_mesh_config,
    summary_for_config,
    generate_icosahedral_mesh,
    get_edge_length_statistics,
    
    # Weight loading and analysis
    load_graphcast_params,
    compute_rigorous_level_energy,
    analyze_processor_weights,
    analyze_encoder_decoder_weights,
    get_available_checkpoints,
    
    # Entropy and power law
    spectral_entropy,
    fit_power_law,
    interpret_exponent,
    interpret_normalized_entropy,
    
    # Visualization
    plot_energy_spectrum,
    plot_entropy_bars,
    set_publication_style,
)


def main():
    print("=" * 70)
    print("RIGOROUS SPECTRAL ENTROPY ANALYSIS OF GRAPHCAST")
    print("=" * 70)
    print()
    
    # Set up plotting
    set_publication_style()
    
    # =========================================================================
    # Step 1: Show available checkpoints
    # =========================================================================
    print("Step 1: Available GraphCast Checkpoints")
    print("-" * 50)
    for key, info in get_available_checkpoints().items():
        print(f"  {key}: {info['name']} ({info['params_mb']} MB)")
    print()
    
    # =========================================================================
    # Step 2: Verify mesh geometry
    # =========================================================================
    print("Step 2: Verifying Icosahedral Mesh Geometry")
    print("-" * 50)
    
    config = get_mesh_config("0.25deg")
    print(summary_for_config(config))
    print()
    
    # Compute exact edge lengths
    print("Computing exact edge lengths from mesh vertices...")
    stats = get_edge_length_statistics(min_level=2, max_level=6)
    
    print(f"\n{'Level':<8} {'Theoretical':>12} {'Actual Mean':>12} {'Std':>10}")
    print("-" * 45)
    for level in config.levels:
        theoretical = MESH_LEVELS[level].approx_km
        actual = stats[level]['mean']
        std = stats[level]['std']
        print(f"M{level:<7} {theoretical:>12,.0f} {actual:>12,.1f} {std:>10,.1f}")
    print()
    
    # =========================================================================
    # Step 3: Download and load GraphCast weights
    # =========================================================================
    print("Step 3: Loading GraphCast Weights")
    print("-" * 50)
    print("Downloading checkpoint from Google Cloud Storage...")
    print("(This may take a few minutes for the first run)")
    print()
    
    try:
        params = load_graphcast_params("0.25deg", verbose=True)
    except ImportError as e:
        print(f"\nError: {e}")
        print("\nTo download weights, you need gcsfs installed:")
        print("  uv add gcsfs")
        return 1
    except Exception as e:
        print(f"\nError loading weights: {e}")
        return 1
    
    print()
    
    # =========================================================================
    # Step 4: Analyze processor weights
    # =========================================================================
    print("Step 4: Processor Weight Analysis")
    print("-" * 50)
    analysis = analyze_processor_weights(params, verbose=True)
    print()
    
    # =========================================================================
    # Step 5: Encoder/Decoder analysis
    # =========================================================================
    print("Step 5: Encoder/Decoder Weight Analysis")
    print("-" * 50)
    enc_dec = analyze_encoder_decoder_weights(params, verbose=True)
    print()
    
    # =========================================================================
    # Step 6: Rigorous sensitivity analysis
    # =========================================================================
    print("Step 6: Rigorous Sensitivity Analysis")
    print("-" * 50)
    result = compute_rigorous_level_energy(params, method="sensitivity", verbose=True)
    level_energy = result.level_energy
    print()
    
    # =========================================================================
    # Step 7: Power law fitting
    # =========================================================================
    print("Step 7: Power Law Fitting")
    print("-" * 50)
    
    levels_arr = np.array(sorted(level_energy.keys()))
    k = np.array([MESH_LEVELS[l].wavenumber for l in levels_arr])
    E = np.array([level_energy[l] for l in levels_arr])
    
    fit = fit_power_law(k, E)
    
    print(f"Model: E(k) = C × k^(-α)")
    print(f"")
    print(f"Amplitude (C): {fit.amplitude:.4e}")
    print(f"Exponent (α): {fit.exponent:.4f} ± {fit.std_err_exponent:.4f}")
    print(f"R²: {fit.r_squared:.4f}")
    print(f"")
    print(f"Comparison to Kolmogorov (α = 5/3 ≈ 1.667):")
    print(f"  Difference: {fit.exponent - 5/3:.4f}")
    print(f"")
    print("Interpretation:")
    print(interpret_exponent(fit.exponent))
    print()
    
    # =========================================================================
    # Step 8: Spectral entropy
    # =========================================================================
    print("Step 8: Spectral Entropy Calculation")
    print("-" * 50)
    
    entropy_result = spectral_entropy(level_energy)
    
    print(f"Raw Entropy (H_s): {entropy_result.H_raw:.4f} nats")
    print(f"Entropy in bits:   {entropy_result.H_bits:.4f} bits")
    print(f"")
    print(f"Normalized Entropy (H_n): {entropy_result.H_normalized:.4f}")
    print(f"  (Range: 0 = single scale, 1 = uniform)")
    print(f"")
    print(f"Maximum possible entropy: {np.log(entropy_result.n_levels):.4f} nats")
    print(f"Dominant scale: M{entropy_result.dominant_scale}")
    print(f"")
    print("Interpretation:")
    print(interpret_normalized_entropy(entropy_result.H_normalized))
    print()
    
    # =========================================================================
    # Step 9: Final summary table with distance labels
    # =========================================================================
    print("=" * 80)
    print("FINAL RESULTS: Energy by Spatial Scale (Icosahedral Mesh)")
    print("=" * 80)
    print(f"\n{'Level':<8} {'Distance (km)':>15} {'Wavenumber':>15} {'Energy':>15} {'Fraction':>12}")
    print("-" * 70)
    
    total_energy = sum(level_energy.values())
    for level in sorted(level_energy.keys()):
        dist = MESH_LEVELS[level].approx_km
        k_val = MESH_LEVELS[level].wavenumber
        energy = level_energy[level]
        frac = energy / total_energy if total_energy > 0 else 0
        print(f"M{level:<7} {dist:>15,.0f} {k_val:>15.2e} {energy:>15.6f} {frac:>11.2%}")
    
    print("-" * 70)
    print(f"{'Total':<8} {'':<15} {'':<15} {total_energy:>15.6f} {'100.00%':>12}")
    print()
    
    # =========================================================================
    # Step 10: Generate plots
    # =========================================================================
    print("Step 10: Generating Visualizations")
    print("-" * 50)
    
    # Create output directory
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    
    # Plot 1: Energy spectrum
    fig1 = plot_energy_spectrum(
        k, E, fit,
        title="GraphCast Weight Energy Spectrum (Sensitivity Analysis)",
        show_kolmogorov=True,
        figsize=(10, 7)
    )
    fig1.savefig(output_dir / "energy_spectrum.png", dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_dir / 'energy_spectrum.png'}")
    
    # Plot 2: Entropy bars
    fig2 = plot_entropy_bars(
        level_energy,
        entropy_result,
        title="Energy Distribution Across Mesh Levels",
        figsize=(12, 6)
    )
    fig2.savefig(output_dir / "entropy_bars.png", dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_dir / 'entropy_bars.png'}")
    
    # Plot 3: Combined summary
    fig3, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Energy vs distance
    distances = [MESH_LEVELS[l].approx_km for l in sorted(level_energy.keys())]
    energies = [level_energy[l] for l in sorted(level_energy.keys())]
    
    ax1 = axes[0]
    ax1.loglog(distances, energies, 'o-', markersize=10, linewidth=2)
    ax1.set_xlabel('Spatial Scale (km)', fontsize=12)
    ax1.set_ylabel('Energy (Sensitivity)', fontsize=12)
    ax1.set_title('Energy vs Spatial Scale', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.invert_xaxis()  # Larger scales on left
    
    for d, e, l in zip(distances, energies, sorted(level_energy.keys())):
        ax1.annotate(f'M{l}', (d, e), textcoords="offset points", xytext=(5, 5), fontsize=9)
    
    # Right: Energy distribution pie chart
    ax2 = axes[1]
    labels = [f'M{l}\n({MESH_LEVELS[l].approx_km:.0f} km)' for l in sorted(level_energy.keys())]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(level_energy)))
    ax2.pie(energies, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
    ax2.set_title('Energy Distribution by Level', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    fig3.savefig(output_dir / "summary.png", dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_dir / 'summary.png'}")
    
    plt.close('all')
    
    print()
    print("=" * 70)
    print("ANALYSIS COMPLETE!")
    print("=" * 70)
    print(f"""
Key Findings:
  - Power Law Exponent: α = {fit.exponent:.4f}
  - Normalized Entropy: H_n = {entropy_result.H_normalized:.4f}
  - Dominant Scale: M{entropy_result.dominant_scale} (~{MESH_LEVELS[entropy_result.dominant_scale].approx_km:.0f} km)

Output files saved to: {output_dir}
  - energy_spectrum.png
  - entropy_bars.png
  - summary.png
""")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
