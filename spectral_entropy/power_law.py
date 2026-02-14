"""
Power law fitting and comparison with turbulence scaling laws.

This module provides tools for fitting power laws E(k) ~ k^(-α) to
spectral energy distributions and comparing with known scaling regimes
like Kolmogorov turbulence.

Key Turbulence Scaling Laws:
    - Kolmogorov (3D): E(k) ~ k^(-5/3) ≈ k^(-1.67)
    - 2D Enstrophy Cascade: E(k) ~ k^(-3)
    - 2D Inverse Energy Cascade: E(k) ~ k^(-5/3)
    - Batchelor (passive scalar): E(k) ~ k^(-1)

The observed k^(-0.49) scaling in GraphCast suggests:
    - "Weakly correlated" information structure
    - Lower "informational viscosity" than physical turbulence
    - More uniform scale participation

References:
    Kolmogorov, A.N. (1941). "The local structure of turbulence."
    Kraichnan, R.H. (1967). "Inertial ranges in two-dimensional turbulence."
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import stats
from scipy.optimize import curve_fit

from spectral_entropy.mesh import MESH_LEVELS


# Known turbulence exponents for reference
TURBULENCE_EXPONENTS = {
    "kolmogorov_3d": 5/3,           # ≈ 1.667
    "kolmogorov_2d_inverse": 5/3,   # Inverse energy cascade
    "enstrophy_cascade": 3.0,       # 2D direct cascade
    "batchelor": 1.0,               # Passive scalar
    "shock_dominated": 2.0,         # Burgers turbulence
    "graphcast_observed": 0.49,     # Your observation
}


@dataclass
class PowerLawFit:
    """Results from power law fitting.
    
    Model: E(k) = C * k^(-α)
    
    Attributes:
        amplitude: Coefficient C
        exponent: Power law exponent α (positive = decay)
        r_squared: Coefficient of determination R²
        std_err_exponent: Standard error of exponent estimate
        residuals: Fit residuals
        p_value: p-value for significance of fit
        n_points: Number of data points used
    """
    amplitude: float
    exponent: float
    r_squared: float
    std_err_exponent: float
    residuals: np.ndarray
    p_value: float
    n_points: int
    
    def __repr__(self) -> str:
        return (
            f"PowerLawFit(\n"
            f"  E(k) = {self.amplitude:.4f} × k^({-self.exponent:.4f})\n"
            f"  R² = {self.r_squared:.4f}\n"
            f"  α = {self.exponent:.4f} ± {self.std_err_exponent:.4f}\n"
            f"  p-value = {self.p_value:.2e}\n"
            f"  n = {self.n_points}\n"
            f")"
        )
    
    def predict(self, k: np.ndarray) -> np.ndarray:
        """Predict E(k) for given wavenumbers."""
        return self.amplitude * k ** (-self.exponent)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "amplitude": self.amplitude,
            "exponent": self.exponent,
            "r_squared": self.r_squared,
            "std_err_exponent": self.std_err_exponent,
            "p_value": self.p_value,
            "n_points": self.n_points,
        }


def _power_law_func(log_k: np.ndarray, log_C: float, alpha: float) -> np.ndarray:
    """Power law in log-log space: log(E) = log(C) - α*log(k)"""
    return log_C - alpha * log_k


def fit_power_law(
    k: np.ndarray,
    E: np.ndarray,
    weights: Optional[np.ndarray] = None,
    method: str = "ols"
) -> PowerLawFit:
    """
    Fit power law E(k) = C * k^(-α) via log-log linear regression.
    
    Performs linear regression in log-log space:
        log(E) = log(C) - α*log(k)
    
    Args:
        k: Wavenumbers (1/km)
        E: Energy values at each wavenumber
        weights: Optional weights for weighted least squares
        method: Fitting method - "ols" (ordinary), "wls" (weighted)
        
    Returns:
        PowerLawFit dataclass with fit results
        
    Raises:
        ValueError: If arrays have different lengths or contain invalid values
    """
    k = np.array(k, dtype=np.float64)
    E = np.array(E, dtype=np.float64)
    
    if len(k) != len(E):
        raise ValueError(f"k and E must have same length, got {len(k)} and {len(E)}")
    
    # Filter out zeros and negatives
    mask = (k > 0) & (E > 0)
    k_valid = k[mask]
    E_valid = E[mask]
    n_points = len(k_valid)
    
    if n_points < 2:
        raise ValueError("Need at least 2 valid data points for fitting")
    
    # Log transform
    log_k = np.log(k_valid)
    log_E = np.log(E_valid)
    
    # Perform linear regression
    if method == "ols" or weights is None:
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_k, log_E)
    elif method == "wls":
        # Weighted least squares
        w = weights[mask] if weights is not None else np.ones_like(log_k)
        
        # Weighted means
        w_sum = np.sum(w)
        mean_x = np.sum(w * log_k) / w_sum
        mean_y = np.sum(w * log_E) / w_sum
        
        # Weighted covariance and variance
        cov_xy = np.sum(w * (log_k - mean_x) * (log_E - mean_y)) / w_sum
        var_x = np.sum(w * (log_k - mean_x) ** 2) / w_sum
        
        slope = cov_xy / var_x
        intercept = mean_y - slope * mean_x
        
        # R² and other stats
        residuals = log_E - (intercept + slope * log_k)
        ss_res = np.sum(w * residuals ** 2)
        ss_tot = np.sum(w * (log_E - mean_y) ** 2)
        r_value = np.sqrt(1 - ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Standard error (approximate)
        mse = ss_res / (n_points - 2) if n_points > 2 else 0
        std_err = np.sqrt(mse / (var_x * w_sum)) if var_x > 0 else 0
        
        # p-value (approximate, using t-distribution)
        t_stat = slope / std_err if std_err > 0 else float('inf')
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n_points - 2))
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'ols' or 'wls'")
    
    # Extract fit parameters
    # slope = -α, so α = -slope
    exponent = -slope
    amplitude = np.exp(intercept)
    r_squared = r_value ** 2
    
    # Compute residuals in original space
    E_predicted = amplitude * k_valid ** (-exponent)
    residuals_original = E_valid - E_predicted
    
    return PowerLawFit(
        amplitude=amplitude,
        exponent=exponent,
        r_squared=r_squared,
        std_err_exponent=std_err,
        residuals=residuals_original,
        p_value=p_value,
        n_points=n_points,
    )


def fit_power_law_nonlinear(
    k: np.ndarray,
    E: np.ndarray,
    initial_guess: Tuple[float, float] = (1.0, 0.5)
) -> PowerLawFit:
    """
    Fit power law using nonlinear least squares in original space.
    
    This can be more robust than log-log OLS for data with
    heteroscedastic errors.
    
    Args:
        k: Wavenumbers
        E: Energy values
        initial_guess: Initial (C, α) for optimization
        
    Returns:
        PowerLawFit dataclass
    """
    k = np.array(k, dtype=np.float64)
    E = np.array(E, dtype=np.float64)
    
    # Filter valid points
    mask = (k > 0) & (E > 0)
    k_valid = k[mask]
    E_valid = E[mask]
    n_points = len(k_valid)
    
    if n_points < 2:
        raise ValueError("Need at least 2 valid data points")
    
    def power_law(k, C, alpha):
        return C * k ** (-alpha)
    
    try:
        popt, pcov = curve_fit(
            power_law, k_valid, E_valid,
            p0=initial_guess,
            bounds=([0, -10], [np.inf, 10]),
            maxfev=10000
        )
        
        amplitude, exponent = popt
        std_err = np.sqrt(np.diag(pcov))[1]  # Standard error of alpha
        
    except (RuntimeError, ValueError):
        # Fall back to OLS
        return fit_power_law(k, E)
    
    # Compute R² and residuals
    E_predicted = power_law(k_valid, *popt)
    residuals = E_valid - E_predicted
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((E_valid - np.mean(E_valid)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    
    # Approximate p-value
    t_stat = exponent / std_err if std_err > 0 else float('inf')
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n_points - 2))
    
    return PowerLawFit(
        amplitude=amplitude,
        exponent=exponent,
        r_squared=r_squared,
        std_err_exponent=std_err,
        residuals=residuals,
        p_value=p_value,
        n_points=n_points,
    )


def kolmogorov_reference(
    k: np.ndarray,
    normalize_to: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Generate Kolmogorov -5/3 reference spectrum.
    
    Args:
        k: Wavenumbers
        normalize_to: If provided, scale to match these energy values
        
    Returns:
        E(k) ~ k^(-5/3) spectrum
    """
    k = np.array(k, dtype=np.float64)
    E_kolmogorov = k ** (-5/3)
    
    if normalize_to is not None:
        # Scale to match mean energy
        scale = np.mean(normalize_to) / np.mean(E_kolmogorov)
        E_kolmogorov *= scale
    
    return E_kolmogorov


def compare_exponents(
    alpha: float,
    reference: str = "kolmogorov_3d"
) -> Dict[str, float]:
    """
    Compare fitted exponent to known turbulence scalings.
    
    Args:
        alpha: Fitted power law exponent
        reference: Specific reference to compare against
        
    Returns:
        Dictionary with comparison metrics
    """
    ref_alpha = TURBULENCE_EXPONENTS.get(reference, 5/3)
    
    return {
        "fitted": alpha,
        "reference": ref_alpha,
        "difference": alpha - ref_alpha,
        "ratio": alpha / ref_alpha if ref_alpha != 0 else float('inf'),
        "percent_difference": 100 * (alpha - ref_alpha) / ref_alpha if ref_alpha != 0 else float('inf'),
    }


def interpret_exponent(alpha: float) -> str:
    """
    Provide physical interpretation of the power law exponent.
    
    Args:
        alpha: Power law exponent (positive = decay)
        
    Returns:
        Interpretation string
    """
    if alpha < 0:
        return (
            "Negative exponent: Energy INCREASES at small scales. "
            "This is unusual and may indicate data issues or very different physics."
        )
    elif alpha < 0.5:
        return (
            f"Very shallow spectrum (α ≈ {alpha:.2f}): "
            "Near-equipartition across scales. "
            "Low 'informational viscosity' - information persists at small scales."
        )
    elif alpha < 1.0:
        return (
            f"Shallow spectrum (α ≈ {alpha:.2f}): "
            "Weak scale correlation. "
            "More uniform than classical turbulence. "
            "Similar to your GraphCast finding (α ≈ 0.49)."
        )
    elif alpha < 1.5:
        return (
            f"Moderate spectrum (α ≈ {alpha:.2f}): "
            "Between Batchelor (α=1) and Kolmogorov (α=5/3). "
            "Partial decorrelation across scales."
        )
    elif alpha < 2.0:
        return (
            f"Kolmogorov-like spectrum (α ≈ {alpha:.2f}): "
            f"Close to 3D turbulence inertial range (α = 5/3 ≈ {5/3:.2f}). "
            "Strong energy cascade from large to small scales."
        )
    elif alpha < 2.5:
        return (
            f"Steep spectrum (α ≈ {alpha:.2f}): "
            "Steeper than Kolmogorov. "
            "May indicate shock-dominated dynamics or strong dissipation."
        )
    elif alpha < 3.5:
        return (
            f"Very steep spectrum (α ≈ {alpha:.2f}): "
            f"Close to 2D enstrophy cascade (α = 3). "
            "Small scales strongly suppressed."
        )
    else:
        return (
            f"Extremely steep spectrum (α ≈ {alpha:.2f}): "
            "Very rapid energy decay. "
            "Small scales almost completely filtered out."
        )


def compute_spectral_slope_local(
    k: np.ndarray,
    E: np.ndarray,
    window_size: int = 3
) -> np.ndarray:
    """
    Compute local spectral slope (instantaneous exponent).
    
    This shows how the scaling varies across wavenumbers,
    useful for detecting scale-dependent behavior.
    
    Args:
        k: Wavenumbers
        E: Energy values
        window_size: Number of points for local slope calculation
        
    Returns:
        Array of local slopes (same length as input, with NaN padding)
    """
    k = np.array(k, dtype=np.float64)
    E = np.array(E, dtype=np.float64)
    
    n = len(k)
    slopes = np.full(n, np.nan)
    
    if n < window_size:
        return slopes
    
    log_k = np.log(k)
    log_E = np.log(E)
    
    half_window = window_size // 2
    
    for i in range(half_window, n - half_window):
        start = i - half_window
        end = i + half_window + 1
        
        slope, _, _, _, _ = stats.linregress(
            log_k[start:end], log_E[start:end]
        )
        slopes[i] = -slope  # Convert to positive exponent
    
    return slopes


def fit_broken_power_law(
    k: np.ndarray,
    E: np.ndarray,
    n_breaks: int = 1
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Fit broken power law with multiple scaling regimes.
    
    Useful if the spectrum has different slopes at different scales
    (e.g., inertial range vs dissipation range).
    
    Args:
        k: Wavenumbers
        E: Energy values
        n_breaks: Number of break points (default 1 = two regimes)
        
    Returns:
        Dictionary with break points and exponents for each regime
    """
    k = np.array(k, dtype=np.float64)
    E = np.array(E, dtype=np.float64)
    
    n = len(k)
    
    if n_breaks == 0:
        fit = fit_power_law(k, E)
        return {
            "n_regimes": 1,
            "break_points": np.array([]),
            "exponents": np.array([fit.exponent]),
            "r_squared_values": np.array([fit.r_squared]),
        }
    
    # Simple approach: try different break points and find best total R²
    best_result = None
    best_score = -np.inf
    
    # Try each possible break point
    for break_idx in range(2, n - 2):
        k1, E1 = k[:break_idx], E[:break_idx]
        k2, E2 = k[break_idx:], E[break_idx:]
        
        try:
            fit1 = fit_power_law(k1, E1)
            fit2 = fit_power_law(k2, E2)
            
            # Combined score (weighted by number of points)
            n1, n2 = len(k1), len(k2)
            score = (n1 * fit1.r_squared + n2 * fit2.r_squared) / (n1 + n2)
            
            if score > best_score:
                best_score = score
                best_result = {
                    "n_regimes": 2,
                    "break_points": np.array([k[break_idx]]),
                    "break_indices": np.array([break_idx]),
                    "exponents": np.array([fit1.exponent, fit2.exponent]),
                    "r_squared_values": np.array([fit1.r_squared, fit2.r_squared]),
                    "combined_r_squared": score,
                }
        except ValueError:
            continue
    
    if best_result is None:
        # Fall back to single power law
        fit = fit_power_law(k, E)
        return {
            "n_regimes": 1,
            "break_points": np.array([]),
            "exponents": np.array([fit.exponent]),
            "r_squared_values": np.array([fit.r_squared]),
        }
    
    return best_result


def summary_fit(
    k: np.ndarray,
    E: np.ndarray
) -> str:
    """
    Generate a summary string of power law fit results.
    
    Args:
        k: Wavenumbers
        E: Energy values
        
    Returns:
        Formatted summary string
    """
    fit = fit_power_law(k, E)
    
    lines = [
        "Power Law Fit Summary",
        "=" * 40,
        f"Model: E(k) = C × k^(-α)",
        f"",
        f"Amplitude (C): {fit.amplitude:.4e}",
        f"Exponent (α): {fit.exponent:.4f} ± {fit.std_err_exponent:.4f}",
        f"R²: {fit.r_squared:.4f}",
        f"p-value: {fit.p_value:.2e}",
        f"N points: {fit.n_points}",
        f"",
        "Interpretation:",
        interpret_exponent(fit.exponent),
        f"",
        "Comparison to Kolmogorov:",
        f"  Kolmogorov α = {5/3:.4f}",
        f"  Difference: {fit.exponent - 5/3:.4f}",
        f"  Ratio: {fit.exponent / (5/3):.4f}",
    ]
    
    return "\n".join(lines)


if __name__ == "__main__":
    # Demo with synthetic data
    print("Power Law Fitting Demo")
    print("=" * 50)
    
    # Generate synthetic k^(-0.5) data with noise
    np.random.seed(42)
    
    levels = np.array(list(MESH_LEVELS.keys()))
    k = np.array([MESH_LEVELS[l].wavenumber for l in levels])
    
    # True parameters
    C_true = 0.023
    alpha_true = 0.49
    
    E_true = C_true * k ** (-alpha_true)
    E_noisy = E_true * (1 + 0.1 * np.random.randn(len(k)))  # 10% noise
    
    # Fit
    fit = fit_power_law(k, E_noisy)
    print(fit)
    print()
    print("Interpretation:")
    print(interpret_exponent(fit.exponent))
