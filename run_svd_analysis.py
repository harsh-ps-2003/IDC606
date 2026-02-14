#!/usr/bin/env python3
"""
Singular Value Distribution Analysis for GraphCast Models.

This script performs comprehensive SVD analysis on GraphCast weight matrices
to test the hypothesis that well-trained neural networks exhibit heavy-tailed
singular value distributions (Martin & Mahoney, 2019).

Analysis includes:
1. SVD of all weight matrices in each model
2. Power-law tail fitting using MLE (Clauset et al., 2009)
3. Marchenko-Pastur random matrix comparison
4. Comparison across three GraphCast model sizes

Usage:
    uv run python run_svd_analysis.py [--models MODEL1 MODEL2 ...] [--output-dir DIR]
    
Arguments:
    --models: Which models to analyze (default: all three)
              Options: 1deg, 0.25deg, operational
    --output-dir: Directory for output files (default: output/svd/)
    --verbose: Print detailed progress

Output:
    - svd_analysis_{model}.json: Per-model analysis results
    - svd_summary.json: Cross-model comparison
    - Visualization plots (PNG)
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
import warnings

import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from spectral_entropy import (
    load_graphcast_params,
    analyze_all_layers,
    summarize_model_spectrum,
    GRAPHCAST_MODELS,
)
from spectral_entropy.singular_value import (
    ModelSpectralAnalysis,
    LayerSpectralAnalysis,
    compute_empirical_spectral_density,
)


def setup_output_dir(output_dir: str) -> Path:
    """Create output directory if it doesn't exist."""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def analyze_single_model(
    model_key: str,
    output_dir: Path,
    verbose: bool = True
) -> ModelSpectralAnalysis:
    """
    Perform complete SVD analysis on a single GraphCast model.
    
    Args:
        model_key: Model identifier (1deg, 0.25deg, operational)
        output_dir: Directory for output files
        verbose: Print progress
        
    Returns:
        ModelSpectralAnalysis with all results
    """
    model_info = GRAPHCAST_MODELS.get(model_key, {})
    model_name = model_info.get("name", model_key)
    
    print(f"\n{'='*70}")
    print(f"ANALYZING: {model_name}")
    print(f"Resolution: {model_info.get('resolution', 'unknown')}")
    print(f"Pressure Levels: {model_info.get('levels', 'unknown')}")
    print(f"{'='*70}\n")
    
    # Load model
    print("Loading checkpoint...")
    params = load_graphcast_params(model_key, verbose=verbose)
    
    # Perform SVD analysis
    print("\nPerforming SVD analysis on all weight matrices...")
    analysis = analyze_all_layers(
        params.raw_params,
        model_name=model_name,
        fit_powerlaw=True,
        fit_mp=True,
        verbose=verbose
    )
    
    # Print summary
    print(f"\n{summarize_model_spectrum(analysis)}")
    
    # Save results
    output_file = output_dir / f"svd_analysis_{model_key}.json"
    analysis.save_json(output_file)
    print(f"\nResults saved to: {output_file}")
    
    return analysis


def print_layer_details(analysis: ModelSpectralAnalysis, top_n: int = 10):
    """Print details of top layers by various metrics."""
    print(f"\n{'='*70}")
    print("TOP LAYERS BY POWER-LAW EXPONENT (α)")
    print(f"{'='*70}")
    
    # Filter layers with good power-law fits
    layers_with_alpha = [
        la for la in analysis.layer_analyses
        if la.powerlaw_fit and la.powerlaw_fit.is_good_fit()
    ]
    
    if not layers_with_alpha:
        print("No layers with valid power-law fits.")
        return
    
    # Sort by alpha
    sorted_by_alpha = sorted(layers_with_alpha, key=lambda x: x.powerlaw_fit.alpha)
    
    print(f"\n{'Layer Name':<50} {'α':>8} {'p-value':>10} {'n_tail':>8}")
    print("-" * 80)
    
    # Heaviest tails (smallest alpha)
    print("\nHeaviest Tails (smallest α):")
    for la in sorted_by_alpha[:top_n]:
        pf = la.powerlaw_fit
        name = la.layer_name[:47] + "..." if len(la.layer_name) > 50 else la.layer_name
        print(f"{name:<50} {pf.alpha:>8.3f} {pf.p_value:>10.4f} {pf.n_tail:>8}")
    
    # Lightest tails (largest alpha)
    print("\nLightest Tails (largest α):")
    for la in sorted_by_alpha[-top_n:]:
        pf = la.powerlaw_fit
        name = la.layer_name[:47] + "..." if len(la.layer_name) > 50 else la.layer_name
        print(f"{name:<50} {pf.alpha:>8.3f} {pf.p_value:>10.4f} {pf.n_tail:>8}")


def analyze_by_component(analysis: ModelSpectralAnalysis):
    """Analyze alpha distribution by model component."""
    print(f"\n{'='*70}")
    print("ANALYSIS BY COMPONENT")
    print(f"{'='*70}")
    
    components = {
        "Encoder (Grid2Mesh)": analysis.get_encoder_layers(),
        "Processor": analysis.get_processor_layers(),
        "Decoder (Mesh2Grid)": analysis.get_decoder_layers(),
    }
    
    print(f"\n{'Component':<25} {'N Layers':>10} {'Mean α':>10} {'Std α':>10} {'Min α':>10} {'Max α':>10}")
    print("-" * 80)
    
    for name, layers in components.items():
        alphas = [
            la.powerlaw_fit.alpha
            for la in layers
            if la.powerlaw_fit and la.powerlaw_fit.is_good_fit()
        ]
        
        if alphas:
            print(f"{name:<25} {len(alphas):>10} {np.mean(alphas):>10.3f} "
                  f"{np.std(alphas):>10.3f} {np.min(alphas):>10.3f} {np.max(alphas):>10.3f}")
        else:
            print(f"{name:<25} {0:>10} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>10}")


def plot_alpha_distribution(
    analyses: dict,
    output_dir: Path
):
    """Plot distribution of alpha values across models."""
    fig, axes = plt.subplots(1, len(analyses), figsize=(5*len(analyses), 5))
    if len(analyses) == 1:
        axes = [axes]
    
    for ax, (model_key, analysis) in zip(axes, analyses.items()):
        alphas = [
            la.powerlaw_fit.alpha
            for la in analysis.layer_analyses
            if la.powerlaw_fit and la.powerlaw_fit.is_good_fit()
        ]
        
        if alphas:
            ax.hist(alphas, bins=30, edgecolor='black', alpha=0.7)
            ax.axvline(np.mean(alphas), color='red', linestyle='--', 
                      label=f'Mean: {np.mean(alphas):.2f}')
            ax.axvline(np.median(alphas), color='green', linestyle=':', 
                      label=f'Median: {np.median(alphas):.2f}')
            
            # Reference lines
            ax.axvline(2, color='orange', linestyle='-', alpha=0.5, label='α=2 (heavy)')
            ax.axvline(4, color='purple', linestyle='-', alpha=0.5, label='α=4 (moderate)')
            
            ax.set_xlabel('Power-Law Exponent (α)')
            ax.set_ylabel('Count')
            ax.set_title(f'{GRAPHCAST_MODELS[model_key]["name"]}\n(N={len(alphas)} layers)')
            ax.legend(fontsize=8)
    
    plt.tight_layout()
    output_file = output_dir / "alpha_distribution.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")


def plot_singular_value_spectrum(
    analysis: ModelSpectralAnalysis,
    output_dir: Path,
    n_layers: int = 6
):
    """Plot singular value spectra for selected layers."""
    # Select diverse layers
    layers = analysis.layer_analyses
    
    # Get encoder, processor, decoder samples
    encoder_layers = [la for la in layers if "grid2mesh" in la.layer_name.lower() or "encoder" in la.layer_name.lower()]
    processor_layers = [la for la in layers if "processor" in la.layer_name.lower()]
    decoder_layers = [la for la in layers if "mesh2grid" in la.layer_name.lower() or "decoder" in la.layer_name.lower()]
    
    selected = []
    for group in [encoder_layers, processor_layers, decoder_layers]:
        if group:
            # Pick layers with different alpha values
            sorted_group = sorted(
                [la for la in group if la.powerlaw_fit and la.powerlaw_fit.is_good_fit()],
                key=lambda x: x.powerlaw_fit.alpha
            )
            if sorted_group:
                selected.append(sorted_group[0])  # Smallest alpha
                if len(sorted_group) > 1:
                    selected.append(sorted_group[-1])  # Largest alpha
    
    if not selected:
        print("No suitable layers for spectrum plot")
        return
    
    n_plots = min(n_layers, len(selected))
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, la in enumerate(selected[:n_plots]):
        ax = axes[i]
        
        sv = la.singular_values
        ranks = np.arange(1, len(sv) + 1)
        
        # Plot singular values
        ax.loglog(ranks, sv, 'b-', linewidth=1, alpha=0.7, label='Singular values')
        
        # Plot power-law fit if available
        if la.powerlaw_fit and la.powerlaw_fit.is_good_fit():
            pf = la.powerlaw_fit
            xmin_idx = np.searchsorted(-sv, -pf.xmin)
            if xmin_idx < len(sv):
                fit_ranks = ranks[xmin_idx:]
                fit_sv = pf.xmin * (fit_ranks / fit_ranks[0]) ** (-1/(pf.alpha - 1))
                ax.loglog(fit_ranks, fit_sv, 'r--', linewidth=2, 
                         label=f'α={pf.alpha:.2f}')
        
        # Formatting
        short_name = la.layer_name.split(":")[-1] if ":" in la.layer_name else la.layer_name
        short_name = short_name[:30] + "..." if len(short_name) > 30 else short_name
        ax.set_title(f'{short_name}\nShape: {la.shape}', fontsize=10)
        ax.set_xlabel('Rank')
        ax.set_ylabel('Singular Value')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Hide unused axes
    for i in range(n_plots, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle(f'Singular Value Spectra: {analysis.model_name}', fontsize=14)
    plt.tight_layout()
    
    output_file = output_dir / f"sv_spectrum_{analysis.model_name.lower().replace(' ', '_')}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")


def plot_alpha_vs_layer_depth(
    analysis: ModelSpectralAnalysis,
    output_dir: Path
):
    """Plot how alpha changes with layer depth in processor."""
    processor_layers = analysis.get_processor_layers()
    
    # Extract step number from layer names
    step_alphas = {}
    for la in processor_layers:
        if la.powerlaw_fit and la.powerlaw_fit.is_good_fit():
            # Try to extract step number
            name = la.layer_name.lower()
            for i in range(16):
                if f"_{i}_" in name or f"_{i}:" in name:
                    if i not in step_alphas:
                        step_alphas[i] = []
                    step_alphas[i].append(la.powerlaw_fit.alpha)
                    break
    
    if not step_alphas:
        print("Could not extract processor step information")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    steps = sorted(step_alphas.keys())
    means = [np.mean(step_alphas[s]) for s in steps]
    stds = [np.std(step_alphas[s]) for s in steps]
    
    ax.errorbar(steps, means, yerr=stds, fmt='o-', capsize=5, capthick=2)
    ax.set_xlabel('Message Passing Step')
    ax.set_ylabel('Mean Power-Law Exponent (α)')
    ax.set_title(f'Alpha vs. Layer Depth: {analysis.model_name}')
    ax.set_xticks(steps)
    ax.grid(True, alpha=0.3)
    
    # Reference lines
    ax.axhline(2, color='orange', linestyle='--', alpha=0.5, label='α=2 (heavy tail)')
    ax.axhline(4, color='purple', linestyle='--', alpha=0.5, label='α=4 (moderate)')
    ax.legend()
    
    plt.tight_layout()
    output_file = output_dir / f"alpha_vs_depth_{analysis.model_name.lower().replace(' ', '_')}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")


def create_summary_comparison(
    analyses: dict,
    output_dir: Path
):
    """Create summary comparison across all models."""
    print(f"\n{'='*70}")
    print("CROSS-MODEL COMPARISON")
    print(f"{'='*70}")
    
    summary = {
        "timestamp": datetime.now().isoformat(),
        "models": {}
    }
    
    print(f"\n{'Model':<25} {'Params':>12} {'N Layers':>10} {'Mean α':>10} {'Weighted α':>12}")
    print("-" * 75)
    
    for model_key, analysis in analyses.items():
        model_name = GRAPHCAST_MODELS[model_key]["name"]
        print(f"{model_name:<25} {analysis.total_params:>12,} {analysis.n_layers:>10} "
              f"{analysis.mean_alpha:>10.3f} {analysis.weighted_alpha:>12.3f}")
        
        summary["models"][model_key] = {
            "name": model_name,
            "total_params": analysis.total_params,
            "n_layers": analysis.n_layers,
            "mean_alpha": analysis.mean_alpha,
            "weighted_alpha": analysis.weighted_alpha,
            "median_alpha": analysis.median_alpha,
            "alpha_std": analysis.alpha_std,
        }
    
    # Theoretical interpretation
    print(f"\n{'='*70}")
    print("THEORETICAL INTERPRETATION")
    print(f"{'='*70}")
    
    for model_key, analysis in analyses.items():
        model_name = GRAPHCAST_MODELS[model_key]["name"]
        alpha = analysis.weighted_alpha
        
        if np.isfinite(alpha):
            # Estimate manifold dimension
            d_estimate = 4.0 / alpha if alpha > 0 else np.inf
            
            print(f"\n{model_name}:")
            print(f"  Weighted α = {alpha:.3f}")
            print(f"  Estimated data manifold dimension d ≈ {d_estimate:.1f}")
            
            if 2 <= alpha <= 4:
                print(f"  Interpretation: Heavy-tailed (good implicit self-regularization)")
            elif alpha < 2:
                print(f"  Interpretation: Very heavy-tailed (may be over-regularized)")
            else:
                print(f"  Interpretation: Light-tailed (may be under-regularized)")
            
            summary["models"][model_key]["manifold_dimension_estimate"] = d_estimate
    
    # Save summary
    def json_serializer(obj):
        """Custom JSON serializer for numpy types."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
    summary_file = output_dir / "svd_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2, default=json_serializer)
    print(f"\nSummary saved to: {summary_file}")
    
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="SVD Analysis of GraphCast Weight Matrices"
    )
    parser.add_argument(
        "--models", 
        nargs="+", 
        default=["1deg", "0.25deg", "operational"],
        choices=["1deg", "0.25deg", "operational"],
        help="Models to analyze"
    )
    parser.add_argument(
        "--output-dir",
        default="output/svd",
        help="Output directory for results"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Print detailed progress"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("SINGULAR VALUE DISTRIBUTION ANALYSIS")
    print("GraphCast Neural Network Weight Matrices")
    print("=" * 70)
    print(f"\nModels to analyze: {args.models}")
    print(f"Output directory: {args.output_dir}")
    
    # Setup
    output_dir = setup_output_dir(args.output_dir)
    
    # Analyze each model
    analyses = {}
    for model_key in args.models:
        try:
            analysis = analyze_single_model(model_key, output_dir, verbose=args.verbose)
            analyses[model_key] = analysis
            
            # Print detailed layer analysis
            print_layer_details(analysis)
            analyze_by_component(analysis)
            
            # Generate per-model plots
            plot_singular_value_spectrum(analysis, output_dir)
            plot_alpha_vs_layer_depth(analysis, output_dir)
            
        except Exception as e:
            print(f"\nERROR analyzing {model_key}: {e}")
            import traceback
            traceback.print_exc()
    
    if not analyses:
        print("\nNo models analyzed successfully!")
        return 1
    
    # Cross-model comparison
    if len(analyses) > 1:
        plot_alpha_distribution(analyses, output_dir)
    
    summary = create_summary_comparison(analyses, output_dir)
    
    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*70}")
    print(f"Results saved to: {output_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
