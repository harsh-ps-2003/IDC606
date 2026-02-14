"""
Spectral Entropy Analysis for GraphCast Neural Network Weights.

This package provides tools to analyze the spectral entropy of multi-mesh
neural network architectures like GraphCast, drawing analogies between
information theory and fluid dynamics turbulence cascades.

Key Components:
    - mesh: Multi-mesh geometry utilities and wavenumber mapping
    - extractor: Weight extraction from GraphCast checkpoints
    - entropy: Shannon and spectral entropy calculations
    - power_law: Power law fitting and Kolmogorov comparisons
    - visualize: Plotting utilities for spectral analysis

Example:
    >>> from spectral_entropy import load_graphcast_params, extract_processor_weights
    >>> from spectral_entropy import spectral_entropy, fit_power_law
    >>> 
    >>> params = load_graphcast_params("0.25deg")
    >>> level_weights = extract_processor_weights(params)
    >>> entropy_result = spectral_entropy(level_weights)
    >>> print(f"Normalized Entropy: {entropy_result['H_normalized']:.4f}")
"""

__version__ = "0.1.0"
__author__ = "IDC606 Research"

# Mesh utilities
from spectral_entropy.mesh import (
    MESH_LEVELS,
    EARTH_CIRCUMFERENCE_KM,
    compute_wavenumber,
    get_edge_length_distribution,
    get_level_spatial_scale,
    map_weights_to_levels,
)

# Weight extraction
from spectral_entropy.extractor import (
    load_graphcast_params,
    extract_processor_weights,
    extract_encoder_decoder_weights,
    get_available_checkpoints,
)

# Entropy calculations
from spectral_entropy.entropy import (
    weight_energy,
    spectral_distribution,
    shannon_entropy,
    normalized_entropy,
    spectral_entropy,
    compute_level_energies,
    interpret_normalized_entropy,
)

# Power law fitting
from spectral_entropy.power_law import (
    PowerLawFit,
    fit_power_law,
    kolmogorov_reference,
    compare_exponents,
    interpret_exponent,
)

# Visualization
from spectral_entropy.visualize import (
    plot_energy_spectrum,
    plot_entropy_bars,
    plot_cascade_diagram,
    plot_level_comparison,
    set_publication_style,
)

__all__ = [
    # Version info
    "__version__",
    "__author__",
    # Mesh
    "MESH_LEVELS",
    "EARTH_CIRCUMFERENCE_KM",
    "compute_wavenumber",
    "get_edge_length_distribution",
    "get_level_spatial_scale",
    "map_weights_to_levels",
    # Extractor
    "load_graphcast_params",
    "extract_processor_weights",
    "extract_encoder_decoder_weights",
    "get_available_checkpoints",
    # Entropy
    "weight_energy",
    "spectral_distribution",
    "shannon_entropy",
    "normalized_entropy",
    "spectral_entropy",
    "compute_level_energies",
    "interpret_normalized_entropy",
    # Power law
    "PowerLawFit",
    "fit_power_law",
    "kolmogorov_reference",
    "compare_exponents",
    "interpret_exponent",
    # Visualization
    "plot_energy_spectrum",
    "plot_entropy_bars",
    "plot_cascade_diagram",
    "plot_level_comparison",
    "set_publication_style",
]
