"""
Visualization utilities for spectral entropy analysis.

This module provides plotting functions for:
    - Log-log energy spectra with power law fits
    - Energy distribution bar charts
    - Cascade diagrams (Sankey-style)
    - Multi-model comparisons

All plots use matplotlib with a clean, publication-ready style.

References:
    Style inspired by: https://arxiv.org/pdf/2212.12794 (GraphCast paper)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.ticker import LogLocator, NullFormatter

from spectral_entropy.mesh import MESH_LEVELS, PHYSICAL_ANALOGS
from spectral_entropy.power_law import PowerLawFit, TURBULENCE_EXPONENTS
from spectral_entropy.entropy import EntropyResult


# Default color palette
COLORS = {
    "data": "#2E8B57",       # Sea green for data points
    "fit": "#DC143C",        # Crimson for fitted lines
    "kolmogorov": "#696969", # Dim gray for reference
    "bars": "#4169E1",       # Royal blue for bars
    "highlight": "#FFD700",  # Gold for highlights
    "grid": "#E0E0E0",       # Light gray for grid
}


def set_publication_style():
    """
    Set matplotlib style for publication-quality figures.
    
    Call this at the start of your analysis to ensure consistent styling.
    """
    plt.rcParams.update({
        # Font settings
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        
        # Figure settings
        "figure.figsize": (8, 6),
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        
        # Axes settings
        "axes.linewidth": 1.0,
        "axes.grid": True,
        "axes.grid.which": "both",
        
        # Grid settings
        "grid.alpha": 0.5,
        "grid.linestyle": "-",
        "grid.linewidth": 0.5,
        
        # Legend settings
        "legend.framealpha": 0.9,
        "legend.edgecolor": "gray",
        
        # Line settings
        "lines.linewidth": 2.0,
        "lines.markersize": 8,
    })


def plot_energy_spectrum(
    k: np.ndarray,
    E: np.ndarray,
    fit: Optional[PowerLawFit] = None,
    show_kolmogorov: bool = True,
    title: str = "Log-Log Power Law Fit: Energy vs Wavenumber",
    xlabel: str = "Wavenumber k (1/km)",
    ylabel: str = r"Energy $E$ ($\Sigma w^2$)",
    save_path: Optional[Union[str, Path]] = None,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (10, 7),
) -> plt.Figure:
    """
    Create log-log plot of energy spectrum with power law fit.
    
    This recreates the style of the image you provided, showing:
    - Data points (green circles)
    - Best fit line (red)
    - Kolmogorov k^(-5/3) reference (gray dashed)
    
    Args:
        k: Wavenumbers (1/km)
        E: Energy values
        fit: Optional PowerLawFit result (will compute if not provided)
        show_kolmogorov: Whether to show Kolmogorov reference line
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        save_path: Path to save figure (if provided)
        ax: Existing axes to plot on (creates new figure if None)
        figsize: Figure size (width, height) in inches
        
    Returns:
        matplotlib Figure object
    """
    from spectral_entropy.power_law import fit_power_law, kolmogorov_reference
    
    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # Fit power law if not provided
    if fit is None:
        fit = fit_power_law(k, E)
    
    # Create extended k range for smooth lines
    k_min, k_max = np.min(k), np.max(k)
    k_line = np.logspace(np.log10(k_min * 0.5), np.log10(k_max * 2), 100)
    
    # Plot data points
    ax.scatter(
        k, E,
        s=100,
        c=COLORS["data"],
        edgecolors="darkgreen",
        linewidths=1.5,
        zorder=10,
        label=f"Data (Weights Energy)"
    )
    
    # Plot fitted power law
    E_fit = fit.predict(k_line)
    ax.plot(
        k_line, E_fit,
        color=COLORS["fit"],
        linewidth=2.5,
        label=f"Best Fit: $E(k) \\sim k^{{-{fit.exponent:.2f}}}$ ($R^2 = {fit.r_squared:.4f}$)"
    )
    
    # Plot Kolmogorov reference
    if show_kolmogorov:
        E_kolmogorov = kolmogorov_reference(k_line, normalize_to=E)
        ax.plot(
            k_line, E_kolmogorov,
            color=COLORS["kolmogorov"],
            linestyle="--",
            linewidth=2.0,
            label=r"Kolmogorov $k^{-5/3}$"
        )
    
    # Set log scale
    ax.set_xscale("log")
    ax.set_yscale("log")
    
    # Labels and title
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    
    # Grid
    ax.grid(True, which="major", linestyle="-", alpha=0.5)
    ax.grid(True, which="minor", linestyle="-", alpha=0.2)
    
    # Legend
    ax.legend(
        loc="lower left",
        framealpha=0.95,
        edgecolor="gray",
        fontsize=10
    )
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if requested
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure to: {save_path}")
    
    return fig


def plot_entropy_bars(
    level_energies: Dict[int, float],
    entropy_result: Optional[EntropyResult] = None,
    title: str = "Energy Distribution Across Mesh Levels",
    save_path: Optional[Union[str, Path]] = None,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (12, 6),
    show_percentages: bool = True,
) -> plt.Figure:
    """
    Create bar chart of energy distribution with entropy annotation.
    
    Args:
        level_energies: Energy at each mesh level
        entropy_result: Optional EntropyResult (will compute if not provided)
        title: Plot title
        save_path: Path to save figure
        ax: Existing axes to plot on
        figsize: Figure size
        show_percentages: Whether to show percentage labels on bars
        
    Returns:
        matplotlib Figure object
    """
    from spectral_entropy.entropy import spectral_entropy as compute_entropy
    
    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # Compute entropy if not provided
    if entropy_result is None:
        entropy_result = compute_entropy(level_energies)
    
    # Prepare data
    levels = sorted(level_energies.keys())
    energies = [level_energies[l] for l in levels]
    total_energy = sum(energies)
    percentages = [100 * e / total_energy for e in energies]
    
    # Create bar chart
    x = np.arange(len(levels))
    bars = ax.bar(
        x, energies,
        color=COLORS["bars"],
        edgecolor="darkblue",
        linewidth=1.5,
        alpha=0.8
    )
    
    # Add percentage labels
    if show_percentages:
        for i, (bar, pct) in enumerate(zip(bars, percentages)):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + max(energies) * 0.02,
                f"{pct:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold"
            )
    
    # X-axis labels with mesh info
    labels = [f"M{l}\n({MESH_LEVELS[l].approx_km:,.0f} km)" for l in levels]
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    
    # Labels
    ax.set_xlabel("Mesh Level (Spatial Scale)", fontsize=12)
    ax.set_ylabel(r"Energy ($\Sigma w^2$)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    
    # Add entropy annotation box
    textstr = (
        f"Spectral Entropy:\n"
        f"  $H_s$ = {entropy_result.H_raw:.4f} nats\n"
        f"  $H_n$ = {entropy_result.H_normalized:.4f}\n"
        f"Dominant: M{entropy_result.dominant_scale}"
    )
    props = dict(boxstyle="round,pad=0.5", facecolor="wheat", alpha=0.9)
    ax.text(
        0.98, 0.98, textstr,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=props,
        family="monospace"
    )
    
    # Grid (y-axis only)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure to: {save_path}")
    
    return fig


def plot_cascade_diagram(
    level_energies: Dict[int, float],
    title: str = "Information Energy Cascade",
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[float, float] = (14, 8),
) -> plt.Figure:
    """
    Create a cascade diagram showing energy flow through mesh levels.
    
    This visualizes the "information cascade" from large to small scales,
    analogous to the energy cascade in turbulence.
    
    Args:
        level_energies: Energy at each mesh level
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Prepare data
    levels = sorted(level_energies.keys())
    energies = [level_energies[l] for l in levels]
    total_energy = sum(energies)
    
    # Normalize to percentages
    pcts = [100 * e / total_energy for e in energies]
    
    # Layout parameters
    n_levels = len(levels)
    x_positions = np.linspace(0.1, 0.9, n_levels)
    box_width = 0.08
    box_height_scale = 0.005  # Scale factor for box height
    y_center = 0.5
    
    # Color gradient (large scale = warm, small scale = cool)
    cmap = plt.cm.RdYlBu_r
    colors = [cmap(i / (n_levels - 1)) for i in range(n_levels)]
    
    # Draw boxes for each level
    for i, (level, pct, color) in enumerate(zip(levels, pcts, colors)):
        x = x_positions[i]
        height = max(pct * box_height_scale, 0.03)  # Minimum height
        
        # Create box
        box = FancyBboxPatch(
            (x - box_width/2, y_center - height/2),
            box_width, height,
            boxstyle="round,pad=0.01,rounding_size=0.02",
            facecolor=color,
            edgecolor="black",
            linewidth=2,
            transform=ax.transAxes,
            zorder=10
        )
        ax.add_patch(box)
        
        # Add level label
        ax.text(
            x, y_center + height/2 + 0.08,
            f"M{level}",
            ha="center", va="bottom",
            fontsize=12, fontweight="bold",
            transform=ax.transAxes
        )
        
        # Add energy percentage
        ax.text(
            x, y_center - height/2 - 0.05,
            f"{pct:.1f}%",
            ha="center", va="top",
            fontsize=10,
            transform=ax.transAxes
        )
        
        # Add scale info
        ax.text(
            x, y_center - height/2 - 0.12,
            f"({MESH_LEVELS[level].approx_km:,.0f} km)",
            ha="center", va="top",
            fontsize=9, color="gray",
            transform=ax.transAxes
        )
    
    # Draw arrows between levels
    for i in range(n_levels - 1):
        x_start = x_positions[i] + box_width/2 + 0.01
        x_end = x_positions[i + 1] - box_width/2 - 0.01
        
        # Arrow thickness proportional to energy transfer
        # (simplified: just show flow direction)
        arrow = FancyArrowPatch(
            (x_start, y_center),
            (x_end, y_center),
            arrowstyle="->,head_length=8,head_width=5",
            color="gray",
            linewidth=2,
            transform=ax.transAxes,
            zorder=5
        )
        ax.add_patch(arrow)
    
    # Add physical analogs at bottom
    for i, level in enumerate(levels):
        analog = PHYSICAL_ANALOGS.get(level, "")
        ax.text(
            x_positions[i], 0.15,
            analog,
            ha="center", va="center",
            fontsize=8, color="dimgray",
            transform=ax.transAxes,
            rotation=45
        )
    
    # Title and labels
    ax.set_title(title, fontsize=16, fontweight="bold", pad=20)
    
    # Add scale labels
    ax.text(0.05, y_center, "Large\nScales", ha="center", va="center",
            fontsize=10, transform=ax.transAxes)
    ax.text(0.95, y_center, "Small\nScales", ha="center", va="center",
            fontsize=10, transform=ax.transAxes)
    
    # Add wavenumber axis annotation
    ax.annotate(
        "", xy=(0.9, 0.25), xytext=(0.1, 0.25),
        arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
        transform=ax.transAxes
    )
    ax.text(0.5, 0.22, "Increasing Wavenumber k", ha="center",
            fontsize=10, transform=ax.transAxes)
    
    # Clean up axes
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure to: {save_path}")
    
    return fig


def plot_level_comparison(
    results: Dict[str, Dict[int, float]],
    metric: str = "energy",
    title: str = "Model Comparison by Mesh Level",
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[float, float] = (12, 6),
) -> plt.Figure:
    """
    Compare multiple models/runs across mesh levels.
    
    Args:
        results: Dict mapping model name to level energies
        metric: What the values represent ("energy", "weights", etc.)
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get all levels
    all_levels = sorted(set(
        level
        for model_data in results.values()
        for level in model_data.keys()
    ))
    
    n_models = len(results)
    n_levels = len(all_levels)
    
    x = np.arange(n_levels)
    width = 0.8 / n_models
    
    # Color cycle
    colors = plt.cm.Set2(np.linspace(0, 1, n_models))
    
    # Plot bars for each model
    for i, (model_name, level_data) in enumerate(results.items()):
        values = [level_data.get(l, 0) for l in all_levels]
        offset = (i - n_models/2 + 0.5) * width
        
        ax.bar(
            x + offset, values,
            width=width * 0.9,
            label=model_name,
            color=colors[i],
            edgecolor="white",
            linewidth=0.5
        )
    
    # Labels
    ax.set_xlabel("Mesh Level", fontsize=12)
    ax.set_ylabel(metric.capitalize(), fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    
    # X-axis
    ax.set_xticks(x)
    ax.set_xticklabels([f"M{l}" for l in all_levels])
    
    # Legend
    ax.legend(loc="upper right")
    
    # Grid
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure to: {save_path}")
    
    return fig


def plot_exponent_comparison(
    fitted_alpha: float,
    fit_error: float = 0.0,
    title: str = "Power Law Exponent Comparison",
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[float, float] = (10, 6),
) -> plt.Figure:
    """
    Compare fitted exponent to known turbulence scalings.
    
    Args:
        fitted_alpha: Your fitted power law exponent
        fit_error: Standard error of the fit
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Reference exponents to show
    references = {
        "GraphCast\n(This work)": fitted_alpha,
        "Kolmogorov\n(3D)": 5/3,
        "Batchelor": 1.0,
        "Enstrophy\nCascade": 3.0,
        "Shock\nDominated": 2.0,
    }
    
    names = list(references.keys())
    values = list(references.values())
    
    x = np.arange(len(names))
    
    # Create bars
    colors = ["#DC143C"] + ["#4169E1"] * (len(names) - 1)  # Highlight our result
    
    bars = ax.bar(x, values, color=colors, edgecolor="black", linewidth=1.5)
    
    # Add error bar for our result
    if fit_error > 0:
        ax.errorbar(0, fitted_alpha, yerr=fit_error, fmt="none",
                   color="black", capsize=5, linewidth=2)
    
    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            f"{val:.3f}",
            ha="center", va="bottom",
            fontsize=10, fontweight="bold"
        )
    
    # Labels
    ax.set_xlabel("Scaling Regime", fontsize=12)
    ax.set_ylabel(r"Exponent $\alpha$", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    
    # Add reference line at Kolmogorov
    ax.axhline(y=5/3, color="gray", linestyle="--", alpha=0.5,
              label=r"Kolmogorov $\alpha = 5/3$")
    
    # Grid
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure to: {save_path}")
    
    return fig


def create_summary_figure(
    k: np.ndarray,
    E: np.ndarray,
    level_energies: Dict[int, float],
    fit: Optional[PowerLawFit] = None,
    entropy_result: Optional[EntropyResult] = None,
    title: str = "GraphCast Spectral Entropy Analysis",
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Create a comprehensive 2x2 summary figure.
    
    Panels:
        (0,0): Log-log energy spectrum
        (0,1): Energy distribution bars
        (1,0): Cascade diagram (simplified)
        (1,1): Entropy decomposition pie chart
    
    Args:
        k: Wavenumbers
        E: Energy values
        level_energies: Energy at each level
        fit: Optional PowerLawFit
        entropy_result: Optional EntropyResult
        title: Overall figure title
        save_path: Path to save figure
        
    Returns:
        matplotlib Figure object
    """
    from spectral_entropy.power_law import fit_power_law
    from spectral_entropy.entropy import spectral_entropy as compute_entropy
    
    # Compute if not provided
    if fit is None:
        fit = fit_power_law(k, E)
    if entropy_result is None:
        entropy_result = compute_entropy(level_energies)
    
    # Create figure
    fig = plt.figure(figsize=(14, 12))
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.98)
    
    # Panel 1: Log-log spectrum (top left)
    ax1 = fig.add_subplot(2, 2, 1)
    plot_energy_spectrum(k, E, fit, ax=ax1, title="Power Law Fit")
    
    # Panel 2: Bar chart (top right)
    ax2 = fig.add_subplot(2, 2, 2)
    plot_entropy_bars(level_energies, entropy_result, ax=ax2,
                     title="Energy Distribution")
    
    # Panel 3: Exponent comparison (bottom left)
    ax3 = fig.add_subplot(2, 2, 3)
    # Simple version - just show key exponents as horizontal bars
    exponents = {
        "This work": fit.exponent,
        "Kolmogorov": 5/3,
        "Batchelor": 1.0,
    }
    y_pos = np.arange(len(exponents))
    colors = ["#DC143C", "#4169E1", "#4169E1"]
    ax3.barh(y_pos, list(exponents.values()), color=colors, edgecolor="black")
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(list(exponents.keys()))
    ax3.set_xlabel(r"Exponent $\alpha$")
    ax3.set_title("Exponent Comparison")
    ax3.axvline(x=5/3, color="gray", linestyle="--", alpha=0.5)
    for i, v in enumerate(exponents.values()):
        ax3.text(v + 0.05, i, f"{v:.3f}", va="center")
    
    # Panel 4: Pie chart of entropy contributions (bottom right)
    ax4 = fig.add_subplot(2, 2, 4)
    levels = sorted(level_energies.keys())
    sizes = [level_energies[l] for l in levels]
    labels = [f"M{l}" for l in levels]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(levels)))
    
    wedges, texts, autotexts = ax4.pie(
        sizes, labels=labels, colors=colors,
        autopct="%1.1f%%", pctdistance=0.7,
        wedgeprops=dict(edgecolor="white", linewidth=2)
    )
    ax4.set_title("Energy Distribution by Level")
    
    # Add entropy annotation
    ax4.text(
        0.5, -0.1,
        f"Normalized Entropy: $H_n$ = {entropy_result.H_normalized:.4f}",
        transform=ax4.transAxes,
        ha="center", fontsize=11, fontweight="bold"
    )
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved summary figure to: {save_path}")
    
    return fig


if __name__ == "__main__":
    # Demo with synthetic data
    print("Visualization Demo")
    print("=" * 50)
    
    set_publication_style()
    
    # Generate synthetic data
    np.random.seed(42)
    
    from spectral_entropy.mesh import MESH_LEVELS
    
    levels = list(MESH_LEVELS.keys())
    k = np.array([MESH_LEVELS[l].wavenumber for l in levels])
    
    # Synthetic energy with k^(-0.5) scaling
    C, alpha = 0.023, 0.49
    E = C * k ** (-alpha) * (1 + 0.1 * np.random.randn(len(k)))
    
    level_energies = {l: E[i] for i, l in enumerate(levels)}
    
    # Create plots
    fig1 = plot_energy_spectrum(k, E, title="Demo: Energy Spectrum")
    plt.show()
    
    fig2 = plot_entropy_bars(level_energies, title="Demo: Energy Distribution")
    plt.show()
    
    print("Demo complete!")
