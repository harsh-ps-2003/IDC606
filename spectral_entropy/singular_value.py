"""
Singular Value Analysis for Neural Network Weight Matrices.

This module implements WeightWatcher-style analysis of neural network weight
matrices, focusing on:
1. Singular value decomposition (SVD) of weight matrices
2. Power-law tail fitting using MLE (following Clauset et al., 2009)
3. Empirical Spectral Density (ESD) computation
4. Marchenko-Pastur random matrix theory baseline comparison

The key insight from Martin & Mahoney (2019) is that well-trained neural
networks exhibit heavy-tailed singular value distributions with power-law
exponents typically in the range [2, 6], indicating implicit self-regularization.

References:
    - Martin & Mahoney (2019): "Traditional and Heavy Tailed Self Regularization
      in Neural Network Models"
    - Clauset, Shalizi & Newman (2009): "Power-law distributions in empirical data"
    - Marchenko & Pastur (1967): "Distribution of eigenvalues for some sets of
      random matrices"
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import warnings
import numpy as np
from scipy import stats
from scipy.optimize import minimize_scalar
import json
from pathlib import Path


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class PowerLawTailFit:
    """
    Results of power-law tail fitting to singular value distribution.
    
    Following Clauset et al. (2009), we fit P(σ) ~ σ^(-α) to the tail
    of the distribution using maximum likelihood estimation.
    
    Attributes:
        alpha: Power-law exponent (typically 2-6 for well-trained NNs)
        xmin: Minimum value for tail fit (cutoff point)
        sigma: Standard error of alpha estimate
        ks_statistic: Kolmogorov-Smirnov goodness-of-fit statistic
        p_value: Statistical significance (p > 0.1 suggests good fit)
        n_tail: Number of data points in the tail (σ >= xmin)
        log_likelihood: Log-likelihood of the fit
    """
    alpha: float
    xmin: float
    sigma: float
    ks_statistic: float
    p_value: float
    n_tail: int
    log_likelihood: float = 0.0
    
    def is_good_fit(self, p_threshold: float = 0.1) -> bool:
        """Check if the power-law fit is statistically acceptable."""
        return self.p_value >= p_threshold and self.n_tail >= 10
    
    def interpret(self) -> str:
        """Interpret the power-law exponent based on Martin & Mahoney."""
        if self.alpha < 2:
            return "Very heavy tail: potentially over-regularized or undertrained"
        elif self.alpha < 4:
            return "Heavy tail: good implicit self-regularization (well-trained)"
        elif self.alpha < 6:
            return "Moderate tail: reasonable regularization"
        else:
            return "Light tail: may indicate under-regularization or overfitting risk"


@dataclass
class MarchenkoPasturFit:
    """
    Marchenko-Pastur distribution fit for random matrix comparison.
    
    The MP distribution describes the eigenvalue distribution of random
    matrices with i.i.d. entries. Deviations from MP indicate learned structure.
    
    Attributes:
        aspect_ratio: Q = rows/cols of the weight matrix
        variance: Estimated variance of matrix entries
        lambda_plus: Upper edge of MP bulk
        lambda_minus: Lower edge of MP bulk
        n_outliers: Number of eigenvalues outside MP bulk
        bulk_fraction: Fraction of eigenvalues within MP bulk
    """
    aspect_ratio: float
    variance: float
    lambda_plus: float
    lambda_minus: float
    n_outliers: int
    bulk_fraction: float
    
    def has_learned_structure(self) -> bool:
        """Check if matrix shows significant deviation from random."""
        return self.n_outliers > 0 or self.bulk_fraction < 0.95


@dataclass
class LayerSpectralAnalysis:
    """
    Complete spectral analysis of a single weight matrix layer.
    
    Attributes:
        layer_name: Identifier for the layer
        shape: Original shape of weight matrix (rows, cols)
        singular_values: Array of singular values in descending order
        rank: Numerical rank (number of singular values > tolerance)
        effective_rank: exp(entropy of normalized singular values)
        condition_number: σ_max / σ_min (numerical stability indicator)
        frobenius_norm: ||W||_F = sqrt(Σσ²)
        spectral_norm: ||W||_2 = σ_max
        stable_rank: ||W||_F² / ||W||_2² (robust rank estimate)
        nuclear_norm: ||W||_* = Σσ (trace norm)
        powerlaw_fit: Power-law tail fit results
        mp_fit: Marchenko-Pastur comparison (optional)
    """
    layer_name: str
    shape: Tuple[int, int]
    singular_values: np.ndarray
    rank: int
    effective_rank: float
    condition_number: float
    frobenius_norm: float
    spectral_norm: float
    stable_rank: float
    nuclear_norm: float
    powerlaw_fit: Optional[PowerLawTailFit] = None
    mp_fit: Optional[MarchenkoPasturFit] = None
    
    @property
    def n_params(self) -> int:
        """Number of parameters in this layer."""
        return self.shape[0] * self.shape[1]
    
    @property
    def compression_ratio(self) -> float:
        """Ratio of effective rank to full rank."""
        return self.effective_rank / min(self.shape)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        # Helper to convert numpy types to Python native types
        def to_python(x):
            if isinstance(x, (np.floating, np.integer)):
                return float(x) if isinstance(x, np.floating) else int(x)
            return x
        
        result = {
            "layer_name": self.layer_name,
            "shape": list(self.shape),
            "n_params": self.n_params,
            "rank": int(self.rank),
            "effective_rank": float(self.effective_rank),
            "condition_number": float(self.condition_number),
            "frobenius_norm": float(self.frobenius_norm),
            "spectral_norm": float(self.spectral_norm),
            "stable_rank": float(self.stable_rank),
            "nuclear_norm": float(self.nuclear_norm),
            "compression_ratio": float(self.compression_ratio),
        }
        if self.powerlaw_fit:
            result["powerlaw"] = {
                "alpha": float(self.powerlaw_fit.alpha),
                "xmin": float(self.powerlaw_fit.xmin),
                "sigma": float(self.powerlaw_fit.sigma),
                "ks_statistic": float(self.powerlaw_fit.ks_statistic),
                "p_value": float(self.powerlaw_fit.p_value),
                "n_tail": int(self.powerlaw_fit.n_tail),
                "is_good_fit": self.powerlaw_fit.is_good_fit(),
            }
        if self.mp_fit:
            result["marchenko_pastur"] = {
                "aspect_ratio": float(self.mp_fit.aspect_ratio),
                "n_outliers": int(self.mp_fit.n_outliers),
                "bulk_fraction": float(self.mp_fit.bulk_fraction),
                "has_learned_structure": self.mp_fit.has_learned_structure(),
            }
        return result


@dataclass
class ModelSpectralAnalysis:
    """
    Complete spectral analysis of all layers in a model.
    
    Attributes:
        model_name: Identifier for the model
        total_params: Total number of parameters
        n_layers: Number of weight matrices analyzed
        layer_analyses: List of per-layer analyses
        mean_alpha: Average power-law exponent (weighted by layer size)
        median_alpha: Median power-law exponent
        alpha_std: Standard deviation of alpha across layers
        total_outliers: Total eigenvalues outside MP bulk
    """
    model_name: str
    total_params: int
    n_layers: int
    layer_analyses: List[LayerSpectralAnalysis]
    mean_alpha: float = 0.0
    median_alpha: float = 0.0
    alpha_std: float = 0.0
    weighted_alpha: float = 0.0
    total_outliers: int = 0
    
    def __post_init__(self):
        """Compute aggregate statistics."""
        alphas = []
        weights = []
        outliers = 0
        
        for la in self.layer_analyses:
            if la.powerlaw_fit and la.powerlaw_fit.is_good_fit():
                alphas.append(la.powerlaw_fit.alpha)
                weights.append(la.n_params)
            if la.mp_fit:
                outliers += la.mp_fit.n_outliers
        
        if alphas:
            self.mean_alpha = np.mean(alphas)
            self.median_alpha = np.median(alphas)
            self.alpha_std = np.std(alphas)
            weights = np.array(weights)
            self.weighted_alpha = np.average(alphas, weights=weights)
        
        self.total_outliers = outliers
    
    def get_layer_by_name(self, name: str) -> Optional[LayerSpectralAnalysis]:
        """Find layer analysis by name."""
        for la in self.layer_analyses:
            if la.layer_name == name:
                return la
        return None
    
    def get_encoder_layers(self) -> List[LayerSpectralAnalysis]:
        """Get all encoder (grid2mesh) layer analyses."""
        return [la for la in self.layer_analyses if "grid2mesh" in la.layer_name.lower() or "encoder" in la.layer_name.lower()]
    
    def get_processor_layers(self) -> List[LayerSpectralAnalysis]:
        """Get all processor layer analyses."""
        return [la for la in self.layer_analyses if "processor" in la.layer_name.lower()]
    
    def get_decoder_layers(self) -> List[LayerSpectralAnalysis]:
        """Get all decoder (mesh2grid) layer analyses."""
        return [la for la in self.layer_analyses if "mesh2grid" in la.layer_name.lower() or "decoder" in la.layer_name.lower()]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "model_name": self.model_name,
            "total_params": int(self.total_params),
            "n_layers": int(self.n_layers),
            "mean_alpha": float(self.mean_alpha) if np.isfinite(self.mean_alpha) else None,
            "median_alpha": float(self.median_alpha) if np.isfinite(self.median_alpha) else None,
            "alpha_std": float(self.alpha_std) if np.isfinite(self.alpha_std) else None,
            "weighted_alpha": float(self.weighted_alpha) if np.isfinite(self.weighted_alpha) else None,
            "total_outliers": int(self.total_outliers),
            "layers": [la.to_dict() for la in self.layer_analyses],
        }
    
    def save_json(self, path: Union[str, Path]) -> None:
        """Save analysis to JSON file."""
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
        
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=json_serializer)


# ============================================================================
# Core SVD Functions
# ============================================================================

def compute_singular_values(W: np.ndarray) -> np.ndarray:
    """
    Compute singular values of a weight matrix.
    
    Args:
        W: Weight matrix of shape (m, n)
        
    Returns:
        Singular values in descending order
    """
    if W.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got shape {W.shape}")
    
    # Use SVD without computing U and V for efficiency
    singular_values = np.linalg.svd(W, compute_uv=False)
    
    return singular_values


def compute_eigenvalues_squared(W: np.ndarray) -> np.ndarray:
    """
    Compute eigenvalues of W^T W (squared singular values).
    
    This is equivalent to σ² and is what the Marchenko-Pastur
    distribution describes.
    
    Args:
        W: Weight matrix of shape (m, n)
        
    Returns:
        Eigenvalues (σ²) in descending order
    """
    sv = compute_singular_values(W)
    return sv ** 2


def compute_effective_rank(singular_values: np.ndarray) -> float:
    """
    Compute effective rank using entropy of normalized singular values.
    
    Effective rank = exp(H) where H = -Σ p_i log(p_i) and p_i = σ_i² / Σσ²
    
    This gives a continuous measure of rank that is more robust than
    counting singular values above a threshold.
    
    Args:
        singular_values: Array of singular values
        
    Returns:
        Effective rank (between 1 and min(m,n))
    """
    # Normalize squared singular values to form probability distribution
    sv_squared = singular_values ** 2
    total = sv_squared.sum()
    
    if total == 0:
        return 0.0
    
    p = sv_squared / total
    
    # Compute entropy, handling zeros
    p_nonzero = p[p > 0]
    entropy = -np.sum(p_nonzero * np.log(p_nonzero))
    
    return np.exp(entropy)


def compute_stable_rank(singular_values: np.ndarray) -> float:
    """
    Compute stable rank = ||W||_F² / ||W||_2².
    
    Stable rank is always between 1 and rank(W), and is more robust
    to small singular values than numerical rank.
    
    Args:
        singular_values: Array of singular values
        
    Returns:
        Stable rank
    """
    if len(singular_values) == 0 or singular_values[0] == 0:
        return 0.0
    
    frobenius_sq = np.sum(singular_values ** 2)
    spectral_sq = singular_values[0] ** 2
    
    return frobenius_sq / spectral_sq


# ============================================================================
# Power-Law Fitting (Clauset et al. methodology)
# ============================================================================

def _powerlaw_log_likelihood(alpha: float, data: np.ndarray, xmin: float) -> float:
    """
    Compute log-likelihood for power-law distribution.
    
    P(x) = (α-1)/xmin * (x/xmin)^(-α)
    
    Args:
        alpha: Power-law exponent
        data: Data points (x >= xmin)
        xmin: Minimum value for power-law
        
    Returns:
        Log-likelihood (negative for minimization)
    """
    if alpha <= 1:
        return np.inf
    
    n = len(data)
    ll = n * np.log(alpha - 1) - n * np.log(xmin) - alpha * np.sum(np.log(data / xmin))
    
    return -ll  # Return negative for minimization


def _estimate_alpha_mle(data: np.ndarray, xmin: float) -> Tuple[float, float]:
    """
    Estimate power-law exponent using MLE.
    
    The MLE for continuous power-law is:
    α = 1 + n / Σ ln(x_i / xmin)
    
    Args:
        data: Data points (x >= xmin)
        xmin: Minimum value
        
    Returns:
        Tuple of (alpha, standard_error)
    """
    n = len(data)
    if n == 0:
        return 0.0, np.inf
    
    log_ratios = np.log(data / xmin)
    sum_log = np.sum(log_ratios)
    
    if sum_log == 0:
        return np.inf, np.inf
    
    alpha = 1 + n / sum_log
    
    # Standard error from Fisher information
    sigma = (alpha - 1) / np.sqrt(n)
    
    return alpha, sigma


def _ks_statistic(data: np.ndarray, alpha: float, xmin: float) -> float:
    """
    Compute Kolmogorov-Smirnov statistic for power-law fit.
    
    Args:
        data: Data points (x >= xmin)
        alpha: Power-law exponent
        xmin: Minimum value
        
    Returns:
        KS statistic (maximum deviation from theoretical CDF)
    """
    n = len(data)
    if n == 0:
        return 1.0
    
    # Sort data
    data_sorted = np.sort(data)
    
    # Empirical CDF
    ecdf = np.arange(1, n + 1) / n
    
    # Theoretical CDF: P(X <= x) = 1 - (x/xmin)^(-(α-1))
    tcdf = 1 - (data_sorted / xmin) ** (-(alpha - 1))
    
    # KS statistic
    ks = np.max(np.abs(ecdf - tcdf))
    
    return ks


def _find_optimal_xmin(data: np.ndarray, xmin_range: Optional[Tuple[float, float]] = None) -> Tuple[float, float, float]:
    """
    Find optimal xmin that minimizes KS statistic.
    
    Args:
        data: Full data array
        xmin_range: Optional range to search (default: use data range)
        
    Returns:
        Tuple of (optimal_xmin, alpha_at_xmin, ks_at_xmin)
    """
    data = np.sort(data)
    
    if xmin_range is None:
        # Search over unique data values in upper 90%
        candidates = np.unique(data)
        candidates = candidates[candidates >= np.percentile(data, 10)]
        candidates = candidates[candidates <= np.percentile(data, 90)]
    else:
        candidates = np.linspace(xmin_range[0], xmin_range[1], 50)
    
    if len(candidates) == 0:
        candidates = [np.median(data)]
    
    best_xmin = candidates[0]
    best_ks = np.inf
    best_alpha = 2.0
    
    for xmin in candidates:
        tail_data = data[data >= xmin]
        if len(tail_data) < 10:
            continue
        
        alpha, _ = _estimate_alpha_mle(tail_data, xmin)
        if alpha <= 1 or not np.isfinite(alpha):
            continue
        
        ks = _ks_statistic(tail_data, alpha, xmin)
        
        if ks < best_ks:
            best_ks = ks
            best_xmin = xmin
            best_alpha = alpha
    
    return best_xmin, best_alpha, best_ks


def fit_powerlaw_tail(
    singular_values: np.ndarray,
    xmin: Union[str, float] = "auto",
    n_bootstrap: int = 100
) -> PowerLawTailFit:
    """
    Fit power-law distribution to the tail of singular values.
    
    Uses the methodology of Clauset, Shalizi & Newman (2009):
    1. Find optimal xmin that minimizes KS statistic
    2. Estimate α using MLE
    3. Compute p-value using bootstrap
    
    Args:
        singular_values: Array of singular values (descending order)
        xmin: Either "auto" to find optimal, or a specific value
        n_bootstrap: Number of bootstrap samples for p-value
        
    Returns:
        PowerLawTailFit with all statistics
    """
    data = singular_values[singular_values > 0]  # Remove zeros
    
    if len(data) < 10:
        return PowerLawTailFit(
            alpha=np.nan, xmin=np.nan, sigma=np.nan,
            ks_statistic=np.nan, p_value=0.0, n_tail=len(data)
        )
    
    # Find optimal xmin or use provided
    if xmin == "auto":
        xmin_opt, alpha_opt, ks_opt = _find_optimal_xmin(data)
    else:
        xmin_opt = float(xmin)
        tail_data = data[data >= xmin_opt]
        alpha_opt, _ = _estimate_alpha_mle(tail_data, xmin_opt)
        ks_opt = _ks_statistic(tail_data, alpha_opt, xmin_opt)
    
    # Get tail data
    tail_data = data[data >= xmin_opt]
    n_tail = len(tail_data)
    
    if n_tail < 10:
        return PowerLawTailFit(
            alpha=alpha_opt, xmin=xmin_opt, sigma=np.nan,
            ks_statistic=ks_opt, p_value=0.0, n_tail=n_tail
        )
    
    # Compute standard error
    _, sigma = _estimate_alpha_mle(tail_data, xmin_opt)
    
    # Compute log-likelihood
    ll = -_powerlaw_log_likelihood(alpha_opt, tail_data, xmin_opt)
    
    # Bootstrap for p-value (simplified)
    # Generate synthetic power-law data and compare KS statistics
    ks_bootstrap = []
    for _ in range(n_bootstrap):
        # Generate power-law samples
        u = np.random.uniform(0, 1, n_tail)
        synthetic = xmin_opt * (1 - u) ** (-1 / (alpha_opt - 1))
        
        # Fit and compute KS
        alpha_syn, _ = _estimate_alpha_mle(synthetic, xmin_opt)
        if alpha_syn > 1:
            ks_syn = _ks_statistic(synthetic, alpha_syn, xmin_opt)
            ks_bootstrap.append(ks_syn)
    
    # p-value: fraction of bootstrap KS >= observed KS
    if ks_bootstrap:
        p_value = np.mean(np.array(ks_bootstrap) >= ks_opt)
    else:
        p_value = 0.0
    
    return PowerLawTailFit(
        alpha=alpha_opt,
        xmin=xmin_opt,
        sigma=sigma,
        ks_statistic=ks_opt,
        p_value=p_value,
        n_tail=n_tail,
        log_likelihood=ll
    )


# ============================================================================
# Marchenko-Pastur Analysis
# ============================================================================

def marchenko_pastur_bounds(aspect_ratio: float, variance: float = 1.0) -> Tuple[float, float]:
    """
    Compute Marchenko-Pastur distribution bounds.
    
    For a random matrix with aspect ratio Q = m/n and entry variance σ²,
    the eigenvalues of (1/m) X^T X are distributed in [λ_-, λ_+].
    
    Args:
        aspect_ratio: Q = rows/cols (Q >= 1)
        variance: Variance of matrix entries
        
    Returns:
        Tuple of (lambda_minus, lambda_plus)
    """
    Q = max(aspect_ratio, 1.0 / aspect_ratio)  # Ensure Q >= 1
    
    lambda_plus = variance * (1 + 1/np.sqrt(Q)) ** 2
    lambda_minus = variance * (1 - 1/np.sqrt(Q)) ** 2
    
    return lambda_minus, lambda_plus


def fit_marchenko_pastur(
    singular_values: np.ndarray,
    shape: Tuple[int, int]
) -> MarchenkoPasturFit:
    """
    Compare singular value distribution to Marchenko-Pastur baseline.
    
    This identifies how much the weight matrix deviates from a random
    matrix, which indicates learned structure.
    
    Args:
        singular_values: Array of singular values
        shape: Original matrix shape (rows, cols)
        
    Returns:
        MarchenkoPasturFit with comparison statistics
    """
    m, n = shape
    Q = m / n if m >= n else n / m
    
    # Estimate variance from bulk of singular values
    # Use median to be robust to outliers
    sv_squared = singular_values ** 2
    
    # Normalize by matrix size
    normalized_sv_sq = sv_squared / max(m, n)
    
    # Estimate variance from the bulk (middle 50%)
    q25, q75 = np.percentile(normalized_sv_sq, [25, 75])
    bulk_mask = (normalized_sv_sq >= q25) & (normalized_sv_sq <= q75)
    
    if np.sum(bulk_mask) > 0:
        variance = np.mean(normalized_sv_sq[bulk_mask])
    else:
        variance = np.median(normalized_sv_sq)
    
    # Compute MP bounds
    lambda_minus, lambda_plus = marchenko_pastur_bounds(Q, variance)
    
    # Count outliers (outside MP bulk)
    # Note: we compare σ² / max(m,n) to MP bounds
    n_outliers = np.sum(normalized_sv_sq > lambda_plus * 1.1)  # 10% tolerance
    
    # Fraction within bulk
    bulk_fraction = np.sum(
        (normalized_sv_sq >= lambda_minus * 0.9) & 
        (normalized_sv_sq <= lambda_plus * 1.1)
    ) / len(singular_values)
    
    return MarchenkoPasturFit(
        aspect_ratio=Q,
        variance=variance,
        lambda_plus=lambda_plus,
        lambda_minus=lambda_minus,
        n_outliers=n_outliers,
        bulk_fraction=bulk_fraction
    )


# ============================================================================
# Empirical Spectral Density
# ============================================================================

@dataclass
class EmpiricalSpectralDensity:
    """
    Empirical spectral density (histogram of singular values).
    
    Attributes:
        bin_centers: Center of each histogram bin
        density: Normalized density in each bin
        bin_edges: Edges of histogram bins
        singular_values: Original singular values
    """
    bin_centers: np.ndarray
    density: np.ndarray
    bin_edges: np.ndarray
    singular_values: np.ndarray


def compute_empirical_spectral_density(
    singular_values: np.ndarray,
    bins: int = 100,
    log_scale: bool = True
) -> EmpiricalSpectralDensity:
    """
    Compute empirical spectral density of singular values.
    
    Args:
        singular_values: Array of singular values
        bins: Number of histogram bins
        log_scale: If True, use logarithmic bins
        
    Returns:
        EmpiricalSpectralDensity with histogram data
    """
    sv = singular_values[singular_values > 0]
    
    if log_scale:
        log_sv = np.log10(sv)
        bin_edges = np.linspace(log_sv.min(), log_sv.max(), bins + 1)
        density, _ = np.histogram(log_sv, bins=bin_edges, density=True)
        bin_centers = 10 ** ((bin_edges[:-1] + bin_edges[1:]) / 2)
        bin_edges = 10 ** bin_edges
    else:
        bin_edges = np.linspace(sv.min(), sv.max(), bins + 1)
        density, _ = np.histogram(sv, bins=bin_edges, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    return EmpiricalSpectralDensity(
        bin_centers=bin_centers,
        density=density,
        bin_edges=bin_edges,
        singular_values=sv
    )


# ============================================================================
# Layer Analysis
# ============================================================================

def analyze_layer_spectrum(
    W: np.ndarray,
    layer_name: str,
    fit_powerlaw: bool = True,
    fit_mp: bool = True,
    rank_tol: float = 1e-10
) -> LayerSpectralAnalysis:
    """
    Perform complete spectral analysis of a single weight matrix.
    
    Args:
        W: Weight matrix of shape (m, n)
        layer_name: Identifier for the layer
        fit_powerlaw: Whether to fit power-law to tail
        fit_mp: Whether to compare to Marchenko-Pastur
        rank_tol: Tolerance for numerical rank computation
        
    Returns:
        LayerSpectralAnalysis with all metrics
    """
    if W.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got shape {W.shape}")
    
    shape = W.shape
    
    # Compute singular values
    sv = compute_singular_values(W)
    
    # Basic metrics
    frobenius_norm = np.sqrt(np.sum(sv ** 2))
    spectral_norm = sv[0] if len(sv) > 0 else 0.0
    nuclear_norm = np.sum(sv)
    
    # Rank metrics
    rank = np.sum(sv > rank_tol * sv[0]) if len(sv) > 0 and sv[0] > 0 else 0
    effective_rank = compute_effective_rank(sv)
    stable_rank = compute_stable_rank(sv)
    
    # Condition number
    if len(sv) > 0 and sv[-1] > 0:
        condition_number = sv[0] / sv[-1]
    else:
        condition_number = np.inf
    
    # Power-law fit
    powerlaw_fit = None
    if fit_powerlaw and len(sv) >= 10:
        try:
            powerlaw_fit = fit_powerlaw_tail(sv)
        except Exception as e:
            warnings.warn(f"Power-law fit failed for {layer_name}: {e}")
    
    # Marchenko-Pastur comparison
    mp_fit = None
    if fit_mp and len(sv) >= 10:
        try:
            mp_fit = fit_marchenko_pastur(sv, shape)
        except Exception as e:
            warnings.warn(f"MP fit failed for {layer_name}: {e}")
    
    return LayerSpectralAnalysis(
        layer_name=layer_name,
        shape=shape,
        singular_values=sv,
        rank=rank,
        effective_rank=effective_rank,
        condition_number=condition_number,
        frobenius_norm=frobenius_norm,
        spectral_norm=spectral_norm,
        stable_rank=stable_rank,
        nuclear_norm=nuclear_norm,
        powerlaw_fit=powerlaw_fit,
        mp_fit=mp_fit
    )


# ============================================================================
# Model-Level Analysis
# ============================================================================

def extract_weight_matrices(params: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Extract all 2D weight matrices from a parameter dictionary.
    
    Args:
        params: Dictionary of parameter arrays (from GraphCast checkpoint)
        
    Returns:
        Dictionary mapping layer names to weight matrices
    """
    weights = {}
    
    for key, value in params.items():
        if isinstance(value, np.ndarray) and value.ndim == 2:
            # Filter to actual weight matrices (not biases)
            # Weight matrices typically have both dimensions > 1
            if value.shape[0] > 1 and value.shape[1] > 1:
                weights[key] = value
    
    return weights


def analyze_all_layers(
    params: Dict[str, np.ndarray],
    model_name: str = "model",
    fit_powerlaw: bool = True,
    fit_mp: bool = True,
    verbose: bool = True
) -> ModelSpectralAnalysis:
    """
    Perform spectral analysis on all weight matrices in a model.
    
    Args:
        params: Dictionary of parameter arrays
        model_name: Identifier for the model
        fit_powerlaw: Whether to fit power-law to each layer
        fit_mp: Whether to compare each layer to Marchenko-Pastur
        verbose: Whether to print progress
        
    Returns:
        ModelSpectralAnalysis with all layer analyses
    """
    # Extract weight matrices
    weights = extract_weight_matrices(params)
    
    if verbose:
        print(f"Analyzing {len(weights)} weight matrices in {model_name}")
    
    # Analyze each layer
    layer_analyses = []
    total_params = 0
    
    for i, (name, W) in enumerate(sorted(weights.items())):
        if verbose and (i + 1) % 10 == 0:
            print(f"  Processing layer {i + 1}/{len(weights)}: {name[:50]}...")
        
        try:
            analysis = analyze_layer_spectrum(
                W, name, 
                fit_powerlaw=fit_powerlaw,
                fit_mp=fit_mp
            )
            layer_analyses.append(analysis)
            total_params += analysis.n_params
        except Exception as e:
            warnings.warn(f"Failed to analyze {name}: {e}")
    
    if verbose:
        print(f"Completed analysis of {len(layer_analyses)} layers")
        print(f"Total parameters: {total_params:,}")
    
    return ModelSpectralAnalysis(
        model_name=model_name,
        total_params=total_params,
        n_layers=len(layer_analyses),
        layer_analyses=layer_analyses
    )


# ============================================================================
# Convenience Functions
# ============================================================================

def quick_svd_analysis(
    W: np.ndarray,
    name: str = "weight"
) -> Dict:
    """
    Quick SVD analysis returning a simple dictionary.
    
    Args:
        W: Weight matrix
        name: Layer name
        
    Returns:
        Dictionary with key metrics
    """
    analysis = analyze_layer_spectrum(W, name)
    return analysis.to_dict()


def summarize_model_spectrum(analysis: ModelSpectralAnalysis) -> str:
    """
    Generate a text summary of model spectral analysis.
    
    Args:
        analysis: ModelSpectralAnalysis object
        
    Returns:
        Formatted string summary
    """
    lines = [
        f"Model Spectral Analysis: {analysis.model_name}",
        "=" * 60,
        f"Total Parameters: {analysis.total_params:,}",
        f"Number of Layers: {analysis.n_layers}",
        "",
        "Power-Law Statistics:",
        f"  Mean α:     {analysis.mean_alpha:.3f}",
        f"  Median α:   {analysis.median_alpha:.3f}",
        f"  Weighted α: {analysis.weighted_alpha:.3f}",
        f"  Std α:      {analysis.alpha_std:.3f}",
        "",
        "Interpretation:",
    ]
    
    if 2 <= analysis.weighted_alpha <= 4:
        lines.append("  → Heavy-tailed: Good implicit self-regularization")
    elif analysis.weighted_alpha < 2:
        lines.append("  → Very heavy-tailed: May be over-regularized")
    else:
        lines.append("  → Light-tailed: May be under-regularized")
    
    lines.extend([
        "",
        f"Marchenko-Pastur Outliers: {analysis.total_outliers}",
        f"  → {'Significant learned structure' if analysis.total_outliers > 10 else 'Moderate learned structure'}",
    ])
    
    return "\n".join(lines)
