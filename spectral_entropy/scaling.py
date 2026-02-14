"""
Neural Network Scaling Law Analysis.

This module implements scaling law analysis for neural networks, examining
how model properties scale with size. Key relationships studied:

1. Loss vs Parameters: L(N) = C * N^(-α)
2. Spectral properties vs Model size
3. Heavy-tail exponent vs Capacity

The theoretical foundation comes from:
- Kaplan et al. (2020): "Scaling Laws for Neural Language Models"
- Hoffmann et al. (2022): "Training Compute-Optimal Large Language Models" (Chinchilla)
- Sharma & Kaplan (2020): "Scaling Laws from the Data Manifold Dimension"

References:
    - Kaplan et al. (2020): Scaling laws for neural language models
    - Hoffmann et al. (2022): Chinchilla scaling laws
    - Bahri et al. (2021): Explaining neural scaling laws
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import warnings
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
import json
from pathlib import Path

from .singular_value import (
    ModelSpectralAnalysis,
    analyze_all_layers,
    summarize_model_spectrum
)
from .extractor import load_graphcast_params, GraphCastParams, CHECKPOINT_INFO


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class ModelMetrics:
    """
    Aggregate metrics for a single model.
    
    Attributes:
        model_name: Identifier for the model
        total_params: Total number of parameters
        n_layers: Number of weight matrices
        mean_alpha: Mean power-law exponent
        weighted_alpha: Size-weighted mean alpha
        median_alpha: Median power-law exponent
        alpha_std: Standard deviation of alpha
        mean_stable_rank: Average stable rank
        mean_effective_rank: Average effective rank
        mean_condition_number: Average condition number (log10)
        total_frobenius: Sum of Frobenius norms
        mp_outliers: Total Marchenko-Pastur outliers
    """
    model_name: str
    total_params: int
    n_layers: int
    mean_alpha: float
    weighted_alpha: float
    median_alpha: float
    alpha_std: float
    mean_stable_rank: float
    mean_effective_rank: float
    mean_condition_number: float
    total_frobenius: float
    mp_outliers: int
    
    # Component-specific metrics
    encoder_alpha: Optional[float] = None
    processor_alpha: Optional[float] = None
    decoder_alpha: Optional[float] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "model_name": self.model_name,
            "total_params": self.total_params,
            "n_layers": self.n_layers,
            "mean_alpha": self.mean_alpha,
            "weighted_alpha": self.weighted_alpha,
            "median_alpha": self.median_alpha,
            "alpha_std": self.alpha_std,
            "mean_stable_rank": self.mean_stable_rank,
            "mean_effective_rank": self.mean_effective_rank,
            "mean_condition_number": self.mean_condition_number,
            "total_frobenius": self.total_frobenius,
            "mp_outliers": self.mp_outliers,
            "encoder_alpha": self.encoder_alpha,
            "processor_alpha": self.processor_alpha,
            "decoder_alpha": self.decoder_alpha,
        }


@dataclass
class ScalingLawFit:
    """
    Results of fitting a scaling law relationship.
    
    Fits the form: y = C * x^(-α) or equivalently log(y) = log(C) - α*log(x)
    
    Attributes:
        exponent: The scaling exponent α
        amplitude: The amplitude C
        r_squared: Coefficient of determination
        std_err: Standard error of the exponent
        n_points: Number of data points used
        x_values: The x values used for fitting
        y_values: The y values used for fitting
        y_predicted: Predicted y values from the fit
    """
    exponent: float
    amplitude: float
    r_squared: float
    std_err: float
    n_points: int
    x_values: np.ndarray
    y_values: np.ndarray
    y_predicted: np.ndarray
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict y values for new x values."""
        return self.amplitude * np.power(x, -self.exponent)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "exponent": self.exponent,
            "amplitude": self.amplitude,
            "r_squared": self.r_squared,
            "std_err": self.std_err,
            "n_points": self.n_points,
        }


@dataclass
class ScalingAnalysis:
    """
    Complete scaling analysis across multiple models.
    
    Attributes:
        model_metrics: List of metrics for each model
        alpha_vs_params: Scaling of alpha with model size
        stable_rank_vs_params: Scaling of stable rank with size
        effective_rank_vs_params: Scaling of effective rank with size
        outliers_vs_params: Scaling of MP outliers with size
    """
    model_metrics: List[ModelMetrics]
    alpha_vs_params: Optional[ScalingLawFit] = None
    stable_rank_vs_params: Optional[ScalingLawFit] = None
    effective_rank_vs_params: Optional[ScalingLawFit] = None
    outliers_vs_params: Optional[ScalingLawFit] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        result = {
            "n_models": len(self.model_metrics),
            "models": [m.to_dict() for m in self.model_metrics],
        }
        if self.alpha_vs_params:
            result["alpha_scaling"] = self.alpha_vs_params.to_dict()
        if self.stable_rank_vs_params:
            result["stable_rank_scaling"] = self.stable_rank_vs_params.to_dict()
        return result
    
    def save_json(self, path: Union[str, Path]) -> None:
        """Save to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


# ============================================================================
# Scaling Law Fitting
# ============================================================================

def fit_scaling_law(
    x: np.ndarray,
    y: np.ndarray,
    weights: Optional[np.ndarray] = None
) -> ScalingLawFit:
    """
    Fit a power-law scaling relationship: y = C * x^(-α).
    
    Uses log-log linear regression for robust fitting.
    
    Args:
        x: Independent variable (e.g., model size)
        y: Dependent variable (e.g., loss or alpha)
        weights: Optional weights for weighted least squares
        
    Returns:
        ScalingLawFit with results
    """
    # Filter valid points
    valid = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    
    if len(x) < 2:
        return ScalingLawFit(
            exponent=np.nan, amplitude=np.nan, r_squared=0.0,
            std_err=np.nan, n_points=len(x),
            x_values=x, y_values=y, y_predicted=np.full_like(y, np.nan)
        )
    
    # Log transform
    log_x = np.log(x)
    log_y = np.log(y)
    
    # Linear regression in log-log space
    if weights is not None:
        weights = weights[valid]
        # Weighted least squares
        W = np.diag(weights)
        X = np.column_stack([np.ones_like(log_x), log_x])
        XtWX = X.T @ W @ X
        XtWy = X.T @ W @ log_y
        try:
            beta = np.linalg.solve(XtWX, XtWy)
        except np.linalg.LinAlgError:
            beta = np.linalg.lstsq(XtWX, XtWy, rcond=None)[0]
        log_C, neg_alpha = beta
    else:
        # Ordinary least squares
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_x, log_y)
        log_C = intercept
        neg_alpha = slope
    
    # Extract parameters
    alpha = -neg_alpha
    C = np.exp(log_C)
    
    # Compute R² and predictions
    y_pred = C * np.power(x, -alpha)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    
    # Standard error (from log-space regression)
    if weights is None:
        residuals = log_y - (log_C + neg_alpha * log_x)
        mse = np.sum(residuals ** 2) / (len(x) - 2) if len(x) > 2 else np.nan
        var_x = np.var(log_x)
        std_err_alpha = np.sqrt(mse / (len(x) * var_x)) if var_x > 0 else np.nan
    else:
        std_err_alpha = np.nan  # Would need more complex calculation
    
    return ScalingLawFit(
        exponent=alpha,
        amplitude=C,
        r_squared=r_squared,
        std_err=std_err_alpha,
        n_points=len(x),
        x_values=x,
        y_values=y,
        y_predicted=y_pred
    )


def fit_linear_scaling(
    x: np.ndarray,
    y: np.ndarray
) -> Tuple[float, float, float]:
    """
    Fit a simple linear relationship: y = a*x + b.
    
    Args:
        x: Independent variable
        y: Dependent variable
        
    Returns:
        Tuple of (slope, intercept, r_squared)
    """
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    
    if len(x) < 2:
        return np.nan, np.nan, 0.0
    
    slope, intercept, r_value, _, _ = stats.linregress(x, y)
    return slope, intercept, r_value ** 2


# ============================================================================
# Model Loading and Metrics Computation
# ============================================================================

# GraphCast model configurations
GRAPHCAST_MODELS = {
    "0.25deg": {
        "name": "GraphCast",
        "resolution": "0.25°",
        "levels": 37,
        "description": "High-resolution model, 37 pressure levels"
    },
    "1deg": {
        "name": "GraphCast_small",
        "resolution": "1°",
        "levels": 13,
        "description": "Small model, 13 pressure levels"
    },
    "operational": {
        "name": "GraphCast_operational",
        "resolution": "0.25°",
        "levels": 13,
        "description": "Operational model, 13 pressure levels"
    }
}


def load_all_models(
    models: Optional[List[str]] = None,
    verbose: bool = True
) -> Dict[str, GraphCastParams]:
    """
    Load multiple GraphCast model checkpoints.
    
    Args:
        models: List of model keys to load (default: all three)
        verbose: Whether to print progress
        
    Returns:
        Dictionary mapping model key to GraphCastParams
    """
    if models is None:
        models = list(GRAPHCAST_MODELS.keys())
    
    loaded = {}
    for model_key in models:
        if model_key not in GRAPHCAST_MODELS:
            warnings.warn(f"Unknown model key: {model_key}")
            continue
        
        if verbose:
            info = GRAPHCAST_MODELS[model_key]
            print(f"\nLoading {info['name']} ({info['resolution']}, {info['levels']} levels)...")
        
        try:
            params = load_graphcast_params(model_key, verbose=verbose)
            loaded[model_key] = params
        except Exception as e:
            warnings.warn(f"Failed to load {model_key}: {e}")
    
    return loaded


def compute_model_metrics(
    spectral_analysis: ModelSpectralAnalysis
) -> ModelMetrics:
    """
    Compute aggregate metrics from spectral analysis.
    
    Args:
        spectral_analysis: ModelSpectralAnalysis object
        
    Returns:
        ModelMetrics with aggregate statistics
    """
    layers = spectral_analysis.layer_analyses
    
    # Collect per-layer statistics
    alphas = []
    alpha_weights = []
    stable_ranks = []
    effective_ranks = []
    condition_numbers = []
    frobenius_norms = []
    
    for la in layers:
        if la.powerlaw_fit and la.powerlaw_fit.is_good_fit():
            alphas.append(la.powerlaw_fit.alpha)
            alpha_weights.append(la.n_params)
        
        stable_ranks.append(la.stable_rank)
        effective_ranks.append(la.effective_rank)
        
        if np.isfinite(la.condition_number) and la.condition_number < 1e15:
            condition_numbers.append(np.log10(la.condition_number))
        
        frobenius_norms.append(la.frobenius_norm)
    
    # Compute aggregates
    mean_alpha = np.mean(alphas) if alphas else np.nan
    median_alpha = np.median(alphas) if alphas else np.nan
    alpha_std = np.std(alphas) if alphas else np.nan
    
    if alphas and alpha_weights:
        weighted_alpha = np.average(alphas, weights=alpha_weights)
    else:
        weighted_alpha = np.nan
    
    # Component-specific alphas
    encoder_alphas = []
    processor_alphas = []
    decoder_alphas = []
    
    for la in layers:
        if not la.powerlaw_fit or not la.powerlaw_fit.is_good_fit():
            continue
        
        name_lower = la.layer_name.lower()
        if "grid2mesh" in name_lower or "encoder" in name_lower:
            encoder_alphas.append(la.powerlaw_fit.alpha)
        elif "mesh2grid" in name_lower or "decoder" in name_lower:
            decoder_alphas.append(la.powerlaw_fit.alpha)
        elif "processor" in name_lower:
            processor_alphas.append(la.powerlaw_fit.alpha)
    
    return ModelMetrics(
        model_name=spectral_analysis.model_name,
        total_params=spectral_analysis.total_params,
        n_layers=spectral_analysis.n_layers,
        mean_alpha=mean_alpha,
        weighted_alpha=weighted_alpha,
        median_alpha=median_alpha,
        alpha_std=alpha_std,
        mean_stable_rank=np.mean(stable_ranks) if stable_ranks else np.nan,
        mean_effective_rank=np.mean(effective_ranks) if effective_ranks else np.nan,
        mean_condition_number=np.mean(condition_numbers) if condition_numbers else np.nan,
        total_frobenius=np.sum(frobenius_norms),
        mp_outliers=spectral_analysis.total_outliers,
        encoder_alpha=np.mean(encoder_alphas) if encoder_alphas else None,
        processor_alpha=np.mean(processor_alphas) if processor_alphas else None,
        decoder_alpha=np.mean(decoder_alphas) if decoder_alphas else None,
    )


# ============================================================================
# Multi-Model Comparison
# ============================================================================

def compare_models(
    models: Dict[str, GraphCastParams],
    verbose: bool = True
) -> ScalingAnalysis:
    """
    Perform scaling analysis across multiple models.
    
    Args:
        models: Dictionary mapping model key to GraphCastParams
        verbose: Whether to print progress
        
    Returns:
        ScalingAnalysis with all comparisons
    """
    # Analyze each model
    spectral_analyses = {}
    metrics_list = []
    
    for model_key, params in models.items():
        if verbose:
            print(f"\n{'='*60}")
            print(f"Analyzing {model_key}...")
            print(f"{'='*60}")
        
        # Get raw parameters
        raw_params = params.raw_params
        
        # Perform spectral analysis
        model_name = GRAPHCAST_MODELS.get(model_key, {}).get("name", model_key)
        spectral = analyze_all_layers(raw_params, model_name=model_name, verbose=verbose)
        spectral_analyses[model_key] = spectral
        
        # Compute metrics
        metrics = compute_model_metrics(spectral)
        metrics_list.append(metrics)
        
        if verbose:
            print(f"\n{summarize_model_spectrum(spectral)}")
    
    # Fit scaling relationships if we have enough models
    alpha_scaling = None
    stable_rank_scaling = None
    effective_rank_scaling = None
    outliers_scaling = None
    
    if len(metrics_list) >= 2:
        params_array = np.array([m.total_params for m in metrics_list])
        
        # Alpha vs params
        alphas = np.array([m.weighted_alpha for m in metrics_list])
        if np.all(np.isfinite(alphas)):
            alpha_scaling = fit_scaling_law(params_array, alphas)
        
        # Stable rank vs params
        stable_ranks = np.array([m.mean_stable_rank for m in metrics_list])
        if np.all(np.isfinite(stable_ranks)):
            stable_rank_scaling = fit_scaling_law(params_array, stable_ranks)
        
        # Effective rank vs params
        effective_ranks = np.array([m.mean_effective_rank for m in metrics_list])
        if np.all(np.isfinite(effective_ranks)):
            effective_rank_scaling = fit_scaling_law(params_array, effective_ranks)
        
        # Outliers vs params
        outliers = np.array([m.mp_outliers for m in metrics_list], dtype=float)
        if np.all(outliers > 0):
            outliers_scaling = fit_scaling_law(params_array, outliers)
    
    return ScalingAnalysis(
        model_metrics=metrics_list,
        alpha_vs_params=alpha_scaling,
        stable_rank_vs_params=stable_rank_scaling,
        effective_rank_vs_params=effective_rank_scaling,
        outliers_vs_params=outliers_scaling,
    )


# ============================================================================
# Theoretical Predictions
# ============================================================================

def predict_alpha_from_manifold_dimension(d: float) -> float:
    """
    Predict scaling exponent from data manifold dimension.
    
    Based on Sharma & Kaplan (2020): α ≈ 4/d
    
    Args:
        d: Intrinsic dimension of data manifold
        
    Returns:
        Predicted scaling exponent
    """
    return 4.0 / d


def estimate_manifold_dimension_from_alpha(alpha: float) -> float:
    """
    Estimate data manifold dimension from observed scaling exponent.
    
    Inverse of predict_alpha_from_manifold_dimension.
    
    Args:
        alpha: Observed scaling exponent
        
    Returns:
        Estimated intrinsic dimension
    """
    if alpha <= 0:
        return np.inf
    return 4.0 / alpha


def kolmogorov_exponent() -> float:
    """Return Kolmogorov turbulence exponent (5/3)."""
    return 5.0 / 3.0


def compare_to_turbulence(alpha: float) -> str:
    """
    Compare observed exponent to Kolmogorov turbulence.
    
    Args:
        alpha: Observed power-law exponent
        
    Returns:
        Interpretation string
    """
    k53 = kolmogorov_exponent()
    
    if abs(alpha - k53) < 0.2:
        return f"Close to Kolmogorov (5/3 ≈ {k53:.3f}): turbulence-like"
    elif alpha < k53:
        return f"Flatter than Kolmogorov: less scale separation than turbulence"
    else:
        return f"Steeper than Kolmogorov: more scale separation than turbulence"


# ============================================================================
# Summary and Reporting
# ============================================================================

def generate_scaling_report(analysis: ScalingAnalysis) -> str:
    """
    Generate a text report of scaling analysis.
    
    Args:
        analysis: ScalingAnalysis object
        
    Returns:
        Formatted report string
    """
    lines = [
        "=" * 70,
        "NEURAL NETWORK SCALING LAW ANALYSIS",
        "=" * 70,
        "",
        "MODEL COMPARISON",
        "-" * 70,
    ]
    
    # Model summary table
    lines.append(f"{'Model':<25} {'Params':>12} {'α (weighted)':>12} {'Stable Rank':>12}")
    lines.append("-" * 70)
    
    for m in analysis.model_metrics:
        lines.append(
            f"{m.model_name:<25} {m.total_params:>12,} "
            f"{m.weighted_alpha:>12.3f} {m.mean_stable_rank:>12.2f}"
        )
    
    lines.extend(["", "SCALING RELATIONSHIPS", "-" * 70])
    
    # Alpha scaling
    if analysis.alpha_vs_params:
        s = analysis.alpha_vs_params
        lines.append(f"α vs N: α ∝ N^({-s.exponent:.4f}), R² = {s.r_squared:.4f}")
        
        # Interpret
        if s.exponent > 0:
            lines.append("  → Larger models have heavier tails (smaller α)")
        else:
            lines.append("  → Larger models have lighter tails (larger α)")
    
    # Stable rank scaling
    if analysis.stable_rank_vs_params:
        s = analysis.stable_rank_vs_params
        lines.append(f"Stable Rank vs N: SR ∝ N^({-s.exponent:.4f}), R² = {s.r_squared:.4f}")
    
    lines.extend(["", "THEORETICAL CONNECTIONS", "-" * 70])
    
    # Manifold dimension estimates
    for m in analysis.model_metrics:
        if np.isfinite(m.weighted_alpha) and m.weighted_alpha > 0:
            d = estimate_manifold_dimension_from_alpha(m.weighted_alpha)
            lines.append(f"{m.model_name}: Estimated manifold dimension d ≈ {d:.1f}")
    
    # Turbulence comparison
    lines.append("")
    for m in analysis.model_metrics:
        if np.isfinite(m.weighted_alpha):
            comparison = compare_to_turbulence(m.weighted_alpha)
            lines.append(f"{m.model_name}: {comparison}")
    
    lines.extend(["", "=" * 70])
    
    return "\n".join(lines)


# ============================================================================
# Convenience Functions
# ============================================================================

def quick_scaling_analysis(
    model_keys: Optional[List[str]] = None,
    verbose: bool = True
) -> ScalingAnalysis:
    """
    Perform complete scaling analysis with minimal setup.
    
    Args:
        model_keys: List of model keys (default: all three GraphCast models)
        verbose: Whether to print progress
        
    Returns:
        ScalingAnalysis with all results
    """
    if model_keys is None:
        model_keys = ["1deg", "0.25deg", "operational"]
    
    # Load models
    models = load_all_models(model_keys, verbose=verbose)
    
    if len(models) == 0:
        raise RuntimeError("No models loaded successfully")
    
    # Perform analysis
    analysis = compare_models(models, verbose=verbose)
    
    if verbose:
        print("\n" + generate_scaling_report(analysis))
    
    return analysis
