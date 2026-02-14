"""
Spectral Entropy Analysis for GraphCast Neural Network Weights.

This package provides tools to analyze the spectral entropy of multi-mesh
neural network architectures like GraphCast, using rigorous methods that
go beyond simple proportional attribution.

RIGOROUS ANALYSIS APPROACHES:
1. Edge Feature Sensitivity: Analyze MLP response to edge features at each scale
   using E[||W*x||²] = ||W*μ||² + tr(W'W Σ)
2. Encoder/Decoder Analysis: Analyze Grid2Mesh and Mesh2Grid weights
3. Exact Geometry: Use precise icosahedral mesh edge lengths

SINGULAR VALUE ANALYSIS (v0.3.0):
4. SVD Analysis: Compute singular value distributions of weight matrices
5. Power-Law Tail Fitting: Fit P(σ) ~ σ^(-α) following Clauset et al. (2009)
6. Marchenko-Pastur Comparison: Compare to random matrix theory baseline
7. Scaling Law Analysis: Compare metrics across model sizes

Key Components:
    - mesh: Multi-mesh geometry utilities, exact mesh generation
    - extractor: Weight extraction with rigorous sensitivity analysis
    - entropy: Shannon and spectral entropy calculations
    - power_law: Power law fitting and Kolmogorov comparisons
    - visualize: Plotting utilities for spectral analysis
    - singular_value: SVD analysis and heavy-tail fitting (NEW)
    - scaling: Multi-model scaling law analysis (NEW)

Example:
    >>> from spectral_entropy import load_graphcast_params, compute_rigorous_level_energy
    >>> from spectral_entropy import spectral_entropy, fit_power_law
    >>> 
    >>> params = load_graphcast_params("0.25deg")
    >>> result = compute_rigorous_level_energy(params, method="sensitivity")
    >>> entropy_result = spectral_entropy(result.level_energy)
    >>> print(f"Normalized Entropy: {entropy_result.H_normalized:.4f}")
    
    # NEW: Singular value analysis
    >>> from spectral_entropy import analyze_all_layers, quick_scaling_analysis
    >>> spectral = analyze_all_layers(params.raw_params, model_name="GraphCast")
    >>> print(f"Mean alpha: {spectral.mean_alpha:.3f}")
"""

__version__ = "0.3.0"
__author__ = "IDC606 Research"

# Mesh utilities
from spectral_entropy.mesh import (
    # Constants
    MESH_LEVELS,
    EARTH_CIRCUMFERENCE_KM,
    EARTH_RADIUS_KM,
    ICOSAHEDRON_EDGE_LENGTH_RAD,
    PHYSICAL_ANALOGS,
    TOTAL_MULTIMESH_EDGES,
    # Data classes
    MeshLevelInfo,
    GraphCastMeshConfig,
    TriangularMesh,
    # Configurations
    GRAPHCAST_CONFIGS,
    get_mesh_config,
    # Exact mesh generation
    generate_icosahedral_mesh,
    merge_meshes,
    faces_to_edges,
    compute_exact_edge_lengths,
    get_exact_edge_lengths_by_level,
    get_edge_length_statistics,
    # Edge classification
    get_edges_per_level,
    get_edge_indices_by_level,
    classify_edges_by_length,
    classify_edge_by_length,
    get_level_edge_indices,
    # Utilities
    compute_wavenumber,
    get_level_spatial_scale,
    compute_edge_geodesic_length_km,
    compute_edge_geodesic_length_rad,
    geodesic_distance,
    get_level_info_table,
    summary,
    summary_for_config,
)

# Weight extraction and rigorous analysis
from spectral_entropy.extractor import (
    # Data classes
    GraphCastParams,
    ProcessorWeightAnalysis,
    RigorousAnalysisResult,
    # Checkpoint loading
    load_graphcast_params,
    load_checkpoint_from_file,
    load_checkpoint_raw,
    get_available_checkpoints,
    CHECKPOINT_INFO,
    # Weight extraction
    extract_first_layer_weights,
    extract_encoder_decoder_weights,
    # Rigorous analysis (main API)
    compute_rigorous_level_energy,
    compute_level_energy_sensitivity,
    compute_expected_edge_features,
    compute_first_layer_sensitivity,
    # Component analysis
    analyze_processor_weights,
    analyze_encoder_decoder_weights,
    # Utilities
    compute_weight_statistics,
    list_checkpoint_keys,
    get_param_shapes,
    quick_load_and_analyze,
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

# Singular value analysis (NEW in v0.3.0)
from spectral_entropy.singular_value import (
    # Data classes
    PowerLawTailFit,
    MarchenkoPasturFit,
    LayerSpectralAnalysis,
    ModelSpectralAnalysis,
    EmpiricalSpectralDensity,
    # Core SVD functions
    compute_singular_values,
    compute_eigenvalues_squared,
    compute_effective_rank,
    compute_stable_rank,
    # Power-law fitting
    fit_powerlaw_tail,
    # Marchenko-Pastur analysis
    marchenko_pastur_bounds,
    fit_marchenko_pastur,
    # Empirical spectral density
    compute_empirical_spectral_density,
    # Layer and model analysis
    analyze_layer_spectrum,
    analyze_all_layers,
    extract_weight_matrices,
    # Utilities
    quick_svd_analysis,
    summarize_model_spectrum,
)

# Scaling law analysis (NEW in v0.3.0)
from spectral_entropy.scaling import (
    # Data classes
    ModelMetrics,
    ScalingLawFit,
    ScalingAnalysis,
    # Model loading
    GRAPHCAST_MODELS,
    load_all_models,
    # Analysis functions
    compute_model_metrics,
    fit_scaling_law,
    fit_linear_scaling,
    compare_models,
    # Theoretical predictions
    predict_alpha_from_manifold_dimension,
    estimate_manifold_dimension_from_alpha,
    kolmogorov_exponent,
    compare_to_turbulence,
    # Reporting
    generate_scaling_report,
    quick_scaling_analysis,
)

__all__ = [
    # Version info
    "__version__",
    "__author__",
    # Mesh constants
    "MESH_LEVELS",
    "EARTH_CIRCUMFERENCE_KM",
    "EARTH_RADIUS_KM",
    "ICOSAHEDRON_EDGE_LENGTH_RAD",
    "PHYSICAL_ANALOGS",
    "TOTAL_MULTIMESH_EDGES",
    # Mesh data classes
    "MeshLevelInfo",
    "GraphCastMeshConfig",
    "TriangularMesh",
    # Mesh configurations
    "GRAPHCAST_CONFIGS",
    "get_mesh_config",
    # Exact mesh generation
    "generate_icosahedral_mesh",
    "merge_meshes",
    "faces_to_edges",
    "compute_exact_edge_lengths",
    "get_exact_edge_lengths_by_level",
    "get_edge_length_statistics",
    # Edge classification
    "get_edges_per_level",
    "get_edge_indices_by_level",
    "classify_edges_by_length",
    "classify_edge_by_length",
    "get_level_edge_indices",
    # Mesh utilities
    "compute_wavenumber",
    "get_level_spatial_scale",
    "compute_edge_geodesic_length_km",
    "compute_edge_geodesic_length_rad",
    "geodesic_distance",
    "get_level_info_table",
    "summary",
    "summary_for_config",
    # Extractor data classes
    "GraphCastParams",
    "ProcessorWeightAnalysis",
    "RigorousAnalysisResult",
    # Checkpoint loading
    "load_graphcast_params",
    "load_checkpoint_from_file",
    "load_checkpoint_raw",
    "get_available_checkpoints",
    "CHECKPOINT_INFO",
    # Weight extraction
    "extract_first_layer_weights",
    "extract_encoder_decoder_weights",
    # Rigorous analysis (main API)
    "compute_rigorous_level_energy",
    "compute_level_energy_sensitivity",
    "compute_expected_edge_features",
    "compute_first_layer_sensitivity",
    # Component analysis
    "analyze_processor_weights",
    "analyze_encoder_decoder_weights",
    # Extractor utilities
    "compute_weight_statistics",
    "list_checkpoint_keys",
    "get_param_shapes",
    "quick_load_and_analyze",
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
    # Singular value analysis (NEW in v0.3.0)
    "PowerLawTailFit",
    "MarchenkoPasturFit",
    "LayerSpectralAnalysis",
    "ModelSpectralAnalysis",
    "EmpiricalSpectralDensity",
    "compute_singular_values",
    "compute_eigenvalues_squared",
    "compute_effective_rank",
    "compute_stable_rank",
    "fit_powerlaw_tail",
    "marchenko_pastur_bounds",
    "fit_marchenko_pastur",
    "compute_empirical_spectral_density",
    "analyze_layer_spectrum",
    "analyze_all_layers",
    "extract_weight_matrices",
    "quick_svd_analysis",
    "summarize_model_spectrum",
    # Scaling law analysis (NEW in v0.3.0)
    "ModelMetrics",
    "ScalingLawFit",
    "ScalingAnalysis",
    "GRAPHCAST_MODELS",
    "load_all_models",
    "compute_model_metrics",
    "fit_scaling_law",
    "fit_linear_scaling",
    "compare_models",
    "predict_alpha_from_manifold_dimension",
    "estimate_manifold_dimension_from_alpha",
    "kolmogorov_exponent",
    "compare_to_turbulence",
    "generate_scaling_report",
    "quick_scaling_analysis",
]
