"""
Shannon and Spectral Entropy calculations for neural network weights.

This module provides functions to compute various entropy measures for
analyzing the "information cascade" in multi-scale neural networks like
GraphCast.

Key Concepts:
    - Weight Energy: E = Σw² (analogous to turbulent kinetic energy)
    - Spectral Distribution: p_i = E_i / ΣE (energy at scale i)
    - Shannon Entropy: H = -Σ p_i ln(p_i)
    - Normalized Entropy: H_n = H / ln(n) ∈ [0, 1]

The spectral entropy measures how uniformly information is distributed
across spatial scales, with high entropy indicating "broadband participation"
and low entropy indicating concentration at specific scales.

References:
    Shannon, C.E. (1948). "A Mathematical Theory of Communication."
    Verma, M.K. (2019). "Energy Transfers in Fluid Flows."
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from spectral_entropy.mesh import MESH_LEVELS, get_all_wavenumbers


@dataclass
class EntropyResult:
    """Results from spectral entropy computation.
    
    Attributes:
        H_raw: Raw Shannon entropy in nats (natural log)
        H_bits: Shannon entropy in bits (log base 2)
        H_normalized: Normalized entropy ∈ [0, 1]
        distribution: Probability distribution p_i
        energies: Energy at each level E_i
        wavenumbers: Wavenumber k_i for each level
        dominant_scale: Level with maximum energy
        n_levels: Number of levels
    """
    H_raw: float
    H_bits: float
    H_normalized: float
    distribution: np.ndarray
    energies: np.ndarray
    wavenumbers: np.ndarray
    dominant_scale: int
    n_levels: int
    
    def __repr__(self) -> str:
        return (
            f"EntropyResult(\n"
            f"  H_raw={self.H_raw:.4f} nats,\n"
            f"  H_bits={self.H_bits:.4f} bits,\n"
            f"  H_normalized={self.H_normalized:.4f},\n"
            f"  dominant_scale=M{self.dominant_scale},\n"
            f"  n_levels={self.n_levels}\n"
            f")"
        )


def weight_energy(weights: np.ndarray) -> float:
    """
    Compute the "energy" of a weight array as Σw².
    
    This is analogous to kinetic energy in turbulence, where
    the squared velocity field represents energy density.
    
    Args:
        weights: Array of neural network weights
        
    Returns:
        Sum of squared weights (energy)
    """
    if weights.size == 0:
        return 0.0
    return float(np.sum(weights.astype(np.float64) ** 2))


def compute_level_energies(
    level_weights: Dict[int, np.ndarray]
) -> Dict[int, float]:
    """
    Compute weight energy at each mesh level.
    
    Args:
        level_weights: Dictionary mapping level to weight arrays
        
    Returns:
        Dictionary mapping level to energy E_i = Σw²
    """
    return {
        level: weight_energy(weights)
        for level, weights in level_weights.items()
    }


def spectral_distribution(
    level_energies: Union[Dict[int, float], np.ndarray]
) -> np.ndarray:
    """
    Convert level energies to a probability distribution.
    
    The spectral distribution p_i = E_i / ΣE represents the
    fraction of total "information energy" at each scale.
    
    Args:
        level_energies: Energy at each level (dict or array)
        
    Returns:
        Normalized probability distribution p_i
        
    Raises:
        ValueError: If total energy is zero
    """
    if isinstance(level_energies, dict):
        # Sort by level and extract values
        levels = sorted(level_energies.keys())
        energies = np.array([level_energies[l] for l in levels], dtype=np.float64)
    else:
        energies = np.array(level_energies, dtype=np.float64)
    
    total_energy = np.sum(energies)
    
    if total_energy == 0:
        raise ValueError("Total energy is zero; cannot compute distribution")
    
    return energies / total_energy


def shannon_entropy(p: np.ndarray, base: str = "natural") -> float:
    """
    Compute Shannon entropy H = -Σ p_i log(p_i).
    
    Args:
        p: Probability distribution (must sum to 1)
        base: Log base - "natural" (nats), "2" (bits), or "10"
        
    Returns:
        Shannon entropy
        
    Raises:
        ValueError: If p doesn't sum to approximately 1
    """
    p = np.array(p, dtype=np.float64)
    
    # Validate probability distribution
    if not np.isclose(np.sum(p), 1.0, rtol=1e-5):
        raise ValueError(f"Probabilities must sum to 1, got {np.sum(p)}")
    
    # Handle zeros (0 * log(0) = 0 by convention)
    p_nonzero = p[p > 0]
    
    if base == "natural":
        log_fn = np.log
    elif base == "2":
        log_fn = np.log2
    elif base == "10":
        log_fn = np.log10
    else:
        raise ValueError(f"Unknown base '{base}'. Use 'natural', '2', or '10'")
    
    return -float(np.sum(p_nonzero * log_fn(p_nonzero)))


def normalized_entropy(H: float, n: int, base: str = "natural") -> float:
    """
    Compute normalized entropy H_n = H / H_max ∈ [0, 1].
    
    The maximum entropy for n outcomes is log(n), achieved when
    the distribution is uniform.
    
    Args:
        H: Shannon entropy
        n: Number of possible outcomes (levels)
        base: Log base used for H
        
    Returns:
        Normalized entropy in [0, 1]
    """
    if n <= 1:
        return 1.0  # Trivial case
    
    if base == "natural":
        H_max = np.log(n)
    elif base == "2":
        H_max = np.log2(n)
    elif base == "10":
        H_max = np.log10(n)
    else:
        raise ValueError(f"Unknown base '{base}'")
    
    return H / H_max


def spectral_entropy(
    level_weights_or_energies: Union[Dict[int, np.ndarray], Dict[int, float]],
    return_dict: bool = False
) -> Union[EntropyResult, Dict]:
    """
    Compute full spectral entropy analysis for multi-level weights.
    
    This is the main entry point for spectral entropy computation.
    It computes energy at each level, normalizes to a probability
    distribution, and calculates Shannon entropy metrics.
    
    Args:
        level_weights_or_energies: Dictionary mapping mesh level to
            either weight arrays (np.ndarray) or pre-computed energies (float)
        return_dict: If True, return a dict instead of EntropyResult
        
    Returns:
        EntropyResult dataclass (or dict if return_dict=True)
        
    Example:
        >>> from spectral_entropy import spectral_entropy
        >>> level_weights = {0: np.random.randn(100), 1: np.random.randn(200)}
        >>> result = spectral_entropy(level_weights)
        >>> print(f"Normalized entropy: {result.H_normalized:.4f}")
    """
    data = level_weights_or_energies
    
    # Check if we have weights or energies
    sample_value = next(iter(data.values()))
    is_weights = isinstance(sample_value, np.ndarray)
    
    # Compute energies if needed
    if is_weights:
        level_energies = compute_level_energies(data)
    else:
        level_energies = {k: float(v) for k, v in data.items()}
    
    # Sort levels
    levels = sorted(level_energies.keys())
    n_levels = len(levels)
    
    # Get energies and wavenumbers as arrays
    energies = np.array([level_energies[l] for l in levels], dtype=np.float64)
    wavenumbers = np.array([MESH_LEVELS[l].wavenumber for l in levels])
    
    # Compute distribution
    p = spectral_distribution(energies)
    
    # Compute entropy measures
    H_raw = shannon_entropy(p, base="natural")
    H_bits = shannon_entropy(p, base="2")
    H_norm = normalized_entropy(H_raw, n_levels, base="natural")
    
    # Find dominant scale
    dominant_scale = levels[np.argmax(energies)]
    
    result_data = {
        "H_raw": H_raw,
        "H_bits": H_bits,
        "H_normalized": H_norm,
        "distribution": p,
        "energies": energies,
        "wavenumbers": wavenumbers,
        "dominant_scale": dominant_scale,
        "n_levels": n_levels,
    }
    
    if return_dict:
        return result_data
    
    return EntropyResult(**result_data)


def relative_entropy(p: np.ndarray, q: np.ndarray) -> float:
    """
    Compute Kullback-Leibler divergence D_KL(p || q).
    
    This measures how different distribution p is from reference q.
    Useful for comparing weight distributions to theoretical models.
    
    Args:
        p: First probability distribution
        q: Second probability distribution (reference)
        
    Returns:
        KL divergence (non-negative)
    """
    p = np.array(p, dtype=np.float64)
    q = np.array(q, dtype=np.float64)
    
    # Handle zeros
    mask = (p > 0) & (q > 0)
    
    if not np.any(mask):
        return float('inf')
    
    return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))


def uniform_distribution(n: int) -> np.ndarray:
    """
    Generate uniform distribution over n outcomes.
    
    Args:
        n: Number of outcomes
        
    Returns:
        Uniform probability distribution
    """
    return np.ones(n) / n


def power_law_distribution(
    levels: np.ndarray,
    alpha: float,
    normalize: bool = True
) -> np.ndarray:
    """
    Generate power-law distribution p_i ∝ k_i^(-α).
    
    This models energy spectra like Kolmogorov turbulence.
    
    Args:
        levels: Array of level indices
        alpha: Power law exponent
        normalize: Whether to normalize to probability distribution
        
    Returns:
        Power-law distributed values
    """
    wavenumbers = np.array([MESH_LEVELS[l].wavenumber for l in levels])
    values = wavenumbers ** (-alpha)
    
    if normalize:
        return values / np.sum(values)
    return values


def compare_to_uniform(
    level_energies: Dict[int, float]
) -> Dict[str, float]:
    """
    Compare observed distribution to uniform distribution.
    
    Args:
        level_energies: Energy at each level
        
    Returns:
        Dictionary with comparison metrics
    """
    p = spectral_distribution(level_energies)
    n = len(p)
    q_uniform = uniform_distribution(n)
    
    return {
        "kl_divergence": relative_entropy(p, q_uniform),
        "max_deviation": float(np.max(np.abs(p - q_uniform))),
        "l2_distance": float(np.sqrt(np.sum((p - q_uniform) ** 2))),
    }


def compare_to_kolmogorov(
    level_energies: Dict[int, float],
    alpha: float = 5/3
) -> Dict[str, float]:
    """
    Compare observed distribution to Kolmogorov k^(-5/3) scaling.
    
    Args:
        level_energies: Energy at each level
        alpha: Kolmogorov exponent (default 5/3)
        
    Returns:
        Dictionary with comparison metrics
    """
    levels = sorted(level_energies.keys())
    p = spectral_distribution(level_energies)
    q_kolmogorov = power_law_distribution(np.array(levels), alpha)
    
    return {
        "kl_divergence": relative_entropy(p, q_kolmogorov),
        "max_deviation": float(np.max(np.abs(p - q_kolmogorov))),
        "l2_distance": float(np.sqrt(np.sum((p - q_kolmogorov) ** 2))),
    }


def entropy_decomposition(
    level_energies: Dict[int, float]
) -> Dict[str, float]:
    """
    Decompose entropy into contributions from each level.
    
    Returns the individual -p_i ln(p_i) terms, showing how much
    each level contributes to the total entropy.
    
    Args:
        level_energies: Energy at each level
        
    Returns:
        Dictionary mapping level to entropy contribution
    """
    p = spectral_distribution(level_energies)
    levels = sorted(level_energies.keys())
    
    contributions = {}
    for i, level in enumerate(levels):
        if p[i] > 0:
            contributions[level] = -float(p[i] * np.log(p[i]))
        else:
            contributions[level] = 0.0
    
    return contributions


def interpret_normalized_entropy(H_n: float) -> str:
    """
    Provide qualitative interpretation of normalized entropy.
    
    Args:
        H_n: Normalized entropy in [0, 1]
        
    Returns:
        Interpretation string
    """
    if H_n >= 0.95:
        return "Near-uniform: Energy distributed almost equally across all scales"
    elif H_n >= 0.85:
        return "High complexity: Broad spectral participation with some scale preferences"
    elif H_n >= 0.70:
        return "Moderate complexity: Clear scale hierarchy but multiple active scales"
    elif H_n >= 0.50:
        return "Low-moderate complexity: Energy concentrated in a subset of scales"
    elif H_n >= 0.30:
        return "Low complexity: Strong concentration at few scales"
    else:
        return "Very low complexity: Energy dominated by one or two scales"


def summary_statistics(
    level_weights: Dict[int, np.ndarray]
) -> Dict[str, float]:
    """
    Compute comprehensive summary statistics for level weights.
    
    Args:
        level_weights: Dictionary mapping level to weight arrays
        
    Returns:
        Dictionary of summary statistics
    """
    result = spectral_entropy(level_weights)
    level_energies = compute_level_energies(level_weights)
    
    total_energy = sum(level_energies.values())
    total_weights = sum(w.size for w in level_weights.values())
    
    return {
        "total_energy": total_energy,
        "total_weights": total_weights,
        "n_levels": result.n_levels,
        "H_raw_nats": result.H_raw,
        "H_bits": result.H_bits,
        "H_normalized": result.H_normalized,
        "dominant_scale": result.dominant_scale,
        "interpretation": interpret_normalized_entropy(result.H_normalized),
    }


if __name__ == "__main__":
    # Demo with synthetic data matching GraphCast hierarchy
    print("Spectral Entropy Demo")
    print("=" * 50)
    
    # Create synthetic weights with decreasing energy at finer scales
    np.random.seed(42)
    level_weights = {}
    for level in range(7):
        # Energy decreases roughly as k^(-0.5)
        scale = MESH_LEVELS[level].approx_km
        n_weights = MESH_LEVELS[level].edges * 10  # ~10 weights per edge
        std = np.sqrt(scale / 7000)  # Scale variance with spatial scale
        level_weights[level] = np.random.randn(n_weights) * std
    
    # Compute entropy
    result = spectral_entropy(level_weights)
    print(result)
    print()
    print(f"Interpretation: {interpret_normalized_entropy(result.H_normalized)}")
