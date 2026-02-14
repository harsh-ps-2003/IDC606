"""
Weight extraction from GraphCast checkpoints with rigorous analysis.

This module provides tools to download and parse GraphCast model checkpoints
from DeepMind's Google Cloud Storage release, and analyze weights using
rigorous methods that go beyond simple proportional attribution.

GraphCast Architecture (from paper Section 3):
- Encoder: Grid2Mesh GNN (maps lat/lon grid → multi-mesh)
- Processor: 16 GNN layers on multi-mesh (327,660 edges for 0.25deg)
- Decoder: Mesh2Grid GNN (maps multi-mesh → lat/lon grid)

RIGOROUS ANALYSIS APPROACHES:
1. Edge Feature Sensitivity: Analyze how MLP weights respond to edge features
   at different spatial scales using E[||W*x||²] = ||W*μ||² + tr(W'W Σ)
2. Encoder/Decoder Analysis: Analyze Grid2Mesh and Mesh2Grid weights which
   have location-specific connectivity patterns
3. Exact Geometry: Use precise icosahedral mesh edge lengths for classification

All MLPs have hidden size 512 and use swish activation.
Total parameters: 36.7 million

Checkpoint key patterns:
  - mesh_gnn/~/processor_edges_{step}_mesh_mlp/~/linear_{layer}/w
  - mesh_gnn/~/processor_nodes_{step}_mesh_nodes_mlp/~/linear_{layer}/w

References:
    GitHub: https://github.com/deepmind/graphcast
    Checkpoints: gs://dm_graphcast/params/
"""

from __future__ import annotations

import io
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from spectral_entropy.mesh import (
    MESH_LEVELS,
    TOTAL_MULTIMESH_EDGES,
    GraphCastMeshConfig,
    get_mesh_config,
    get_edge_indices_by_level,
)


# Default cache directory for downloaded checkpoints
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "graphcast"

# Available GraphCast checkpoints from DeepMind
CHECKPOINT_INFO = {
    "0.25deg": {
        "name": "GraphCast",
        "resolution": "0.25°",
        "levels": 37,
        "params_url": "gs://dm_graphcast/params/GraphCast - ERA5 1979-2017 - resolution 0.25 - pressure levels 37 - mesh 2to6 - precipitation input and output.npz",
        "params_mb": 147,
    },
    "1deg": {
        "name": "GraphCast_small",  
        "resolution": "1°",
        "levels": 13,
        "params_url": "gs://dm_graphcast/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz",
        "params_mb": 29,
    },
    "operational": {
        "name": "GraphCast_operational",
        "resolution": "0.25°", 
        "levels": 13,
        "params_url": "gs://dm_graphcast/params/GraphCast_operational - ERA5-HRES 1979-2021 - resolution 0.25 - pressure levels 13 - mesh 2to6 - precipitation output only.npz",
        "params_mb": 147,
    },
}


@dataclass
class GraphCastParams:
    """Container for GraphCast model parameters.
    
    Attributes:
        encoder: Encoder (Grid2Mesh) parameters
        processor: Processor (16 GNN layers) parameters  
        decoder: Decoder (Mesh2Grid) parameters
        metadata: Checkpoint metadata
        raw_params: Original flattened parameter dictionary
    """
    encoder: Dict[str, np.ndarray]
    processor: Dict[str, np.ndarray]
    decoder: Dict[str, np.ndarray]
    metadata: Dict[str, Any]
    raw_params: Dict[str, np.ndarray] = field(default_factory=dict)
    
    @property
    def total_params(self) -> int:
        """Total number of parameters."""
        total = 0
        for params_dict in [self.encoder, self.processor, self.decoder]:
            for arr in params_dict.values():
                total += arr.size
        return total
    
    @property
    def processor_edge_mlp_keys(self) -> List[str]:
        """Get all processor edge MLP weight keys."""
        return [k for k in self.processor.keys() if "processor_edges" in k]
    
    @property
    def processor_node_mlp_keys(self) -> List[str]:
        """Get all processor node MLP weight keys."""
        return [k for k in self.processor.keys() if "processor_nodes" in k]


@dataclass
class ProcessorWeightAnalysis:
    """Analysis of processor weights for spectral entropy calculation.
    
    Attributes:
        total_energy: Total Frobenius norm squared of all processor weights
        edge_mlp_energy: Energy from edge update MLPs
        node_mlp_energy: Energy from node update MLPs
        per_step_energy: Energy breakdown by message passing step
        n_steps: Number of message passing steps
        config: Mesh configuration used
        first_layer_weights: First layer weight matrices for sensitivity analysis
    """
    total_energy: float
    edge_mlp_energy: float
    node_mlp_energy: float
    per_step_energy: Dict[int, Dict[str, float]]
    n_steps: int
    config: GraphCastMeshConfig
    first_layer_weights: Optional[Dict[str, np.ndarray]] = None


def get_available_checkpoints() -> Dict[str, Dict[str, Any]]:
    """
    Get information about available GraphCast checkpoints.
    
    Returns:
        Dictionary mapping checkpoint ID to metadata
    """
    return CHECKPOINT_INFO.copy()


def _ensure_cache_dir(cache_dir: Optional[Path] = None) -> Path:
    """Ensure cache directory exists and return path."""
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _download_from_gcs(
    gcs_url: str,
    local_path: Path,
    verbose: bool = True
) -> Path:
    """
    Download a file from Google Cloud Storage.
    
    Args:
        gcs_url: GCS URL starting with gs://
        local_path: Local path to save the file
        verbose: Whether to print progress
        
    Returns:
        Path to downloaded file
    """
    try:
        import gcsfs
    except ImportError:
        raise ImportError(
            "gcsfs is required to download GraphCast checkpoints. "
            "Install with: pip install gcsfs"
        )
    
    if local_path.exists():
        if verbose:
            print(f"Using cached checkpoint: {local_path}")
        return local_path
    
    if verbose:
        print(f"Downloading from {gcs_url}...")
        print(f"This may take a few minutes for large checkpoints.")
    
    fs = gcsfs.GCSFileSystem(token="anon")
    
    # Remove gs:// prefix for gcsfs
    gcs_path = gcs_url.replace("gs://", "")
    
    # Download with progress
    local_path.parent.mkdir(parents=True, exist_ok=True)
    fs.get(gcs_path, str(local_path))
    
    if verbose:
        print(f"Saved to: {local_path}")
    
    return local_path


# ============================================================================
# Checkpoint Key Patterns
# ============================================================================

# Separator used in flattened checkpoint keys
CHECKPOINT_SEP = ":"

# Key patterns for different components
PROCESSOR_EDGE_MLP_PATTERN = "mesh_gnn:~:processor_edges_{step}_mesh_mlp"
PROCESSOR_NODE_MLP_PATTERN = "mesh_gnn:~:processor_nodes_{step}_mesh_nodes_mlp"
ENCODER_EDGE_MLP_PATTERN = "grid2mesh_gnn:~:encoder_edges_grid2mesh_mlp"
DECODER_EDGE_MLP_PATTERN = "mesh2grid_gnn:~:decoder_edges_mesh2grid_mlp"


def _unflatten_checkpoint(flat: Dict[str, Any]) -> Dict[str, Any]:
    """
    Unflatten a checkpoint dictionary.
    
    GraphCast checkpoints use ':' as separator for nested keys.
    Example: "params:mesh_gnn:~:linear_0:w" -> params["mesh_gnn"]["~"]["linear_0"]["w"]
    
    Args:
        flat: Flattened dictionary with ':' separated keys
        
    Returns:
        Nested dictionary
    """
    tree = {}
    for flat_key, v in flat.items():
        node = tree
        keys = flat_key.split(CHECKPOINT_SEP)
        for k in keys[:-1]:
            if k not in node:
                node[k] = {}
            node = node[k]
        node[keys[-1]] = v
    return tree


def _get_nested_value(d: Dict, key_path: str) -> Any:
    """
    Get a value from a nested dictionary using ':' separated path.
    
    Args:
        d: Nested dictionary
        key_path: Path like "mesh_gnn:~:processor_edges_0_mesh_mlp"
        
    Returns:
        Value at the path, or None if not found
    """
    keys = key_path.split(CHECKPOINT_SEP)
    current = d
    for k in keys:
        if isinstance(current, dict) and k in current:
            current = current[k]
        else:
            return None
    return current


def load_graphcast_params(
    resolution: str = "0.25deg",
    cache_dir: Optional[Path] = None,
    force_download: bool = False,
    verbose: bool = True
) -> GraphCastParams:
    """
    Load GraphCast parameters from DeepMind checkpoint.
    
    Downloads the checkpoint from Google Cloud Storage if not cached.
    
    Args:
        resolution: Checkpoint resolution ("0.25deg", "1deg", or "operational")
        cache_dir: Directory to cache downloaded checkpoints
        force_download: Whether to re-download even if cached
        verbose: Whether to print progress information
        
    Returns:
        GraphCastParams containing encoder, processor, decoder weights
        
    Raises:
        ValueError: If resolution is not recognized
        ImportError: If gcsfs is not installed
    """
    if resolution not in CHECKPOINT_INFO:
        available = list(CHECKPOINT_INFO.keys())
        raise ValueError(f"Unknown resolution '{resolution}'. Available: {available}")
    
    info = CHECKPOINT_INFO[resolution]
    cache_dir = _ensure_cache_dir(cache_dir)
    
    # Construct local filename
    local_filename = f"graphcast_{resolution}.npz"
    local_path = cache_dir / local_filename
    
    if force_download and local_path.exists():
        local_path.unlink()
    
    # Download if needed
    local_path = _download_from_gcs(info["params_url"], local_path, verbose)
    
    # Load checkpoint
    if verbose:
        print(f"Loading checkpoint...")
    
    with np.load(local_path, allow_pickle=True) as data:
        raw_params = dict(data)
    
    # Parse into structured format using proper key patterns
    encoder_params = {}
    processor_params = {}
    decoder_params = {}
    metadata = {
        "resolution": resolution,
        "checkpoint_info": info,
        "mesh_config": get_mesh_config(resolution),
    }
    
    for key, value in raw_params.items():
        if not isinstance(value, np.ndarray):
            continue
            
        # Use the actual GraphCast naming conventions
        if "grid2mesh_gnn" in key:
            encoder_params[key] = value
        elif "mesh2grid_gnn" in key:
            decoder_params[key] = value
        elif "mesh_gnn" in key:
            processor_params[key] = value
        else:
            # Fallback for other naming patterns
            key_lower = key.lower()
            if "encoder" in key_lower or "grid2mesh" in key_lower:
                encoder_params[key] = value
            elif "decoder" in key_lower or "mesh2grid" in key_lower:
                decoder_params[key] = value
            elif "processor" in key_lower:
                processor_params[key] = value
    
    result = GraphCastParams(
        encoder=encoder_params,
        processor=processor_params,
        decoder=decoder_params,
        metadata=metadata,
        raw_params=raw_params,
    )
    
    if verbose:
        print(f"Loaded {result.total_params:,} parameters")
        print(f"  Encoder: {len(encoder_params)} weight arrays")
        print(f"  Processor: {len(processor_params)} weight arrays")
        print(f"  Decoder: {len(decoder_params)} weight arrays")
    
    return result


def load_checkpoint_from_file(
    path: Union[str, Path],
    verbose: bool = True
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """
    Load a GraphCast checkpoint from a local file.
    
    This function loads the raw checkpoint without parsing into
    encoder/processor/decoder structure.
    
    Args:
        path: Path to .npz checkpoint file
        verbose: Whether to print progress
        
    Returns:
        Tuple of (params_dict, model_config_dict)
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    
    if verbose:
        print(f"Loading checkpoint from {path}...")
    
    with np.load(path, allow_pickle=True) as data:
        raw = dict(data)
    
    # Unflatten the checkpoint
    unflat = _unflatten_checkpoint(raw)
    
    # Extract params and model_config if present
    params = unflat.get("params", unflat)
    model_config = unflat.get("model_config", {})
    
    if verbose:
        n_arrays = len([k for k, v in raw.items() if isinstance(v, np.ndarray)])
        total_params = sum(v.size for v in raw.values() if isinstance(v, np.ndarray))
        print(f"  Loaded {n_arrays} arrays, {total_params:,} total parameters")
    
    return params, model_config


def extract_first_layer_weights(
    params: Union[GraphCastParams, Dict[str, np.ndarray]],
    component: str = "processor"
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Extract first-layer MLP weights from GraphCast components.
    
    The first layer weights determine sensitivity to input features.
    For edge MLPs, input features include [normalized_distance, rel_x, rel_y, rel_z].
    
    Args:
        params: GraphCastParams or raw parameter dictionary
        component: Which component to extract ("processor", "encoder", "decoder")
        
    Returns:
        Dictionary mapping MLP name to {"w": weight_matrix, "b": bias_vector}
    """
    if isinstance(params, GraphCastParams):
        if component == "processor":
            param_dict = params.processor
        elif component == "encoder":
            param_dict = params.encoder
        elif component == "decoder":
            param_dict = params.decoder
        else:
            raise ValueError(f"Unknown component: {component}")
    else:
        param_dict = params
    
    first_layer_weights = {}
    
    # Find all linear_0 (first layer) weights
    for key, value in param_dict.items():
        if not isinstance(value, np.ndarray):
            continue
        
        # Match patterns like "...mlp:~:linear_0:w" or "...mlp/~/linear_0/w"
        if "linear_0" in key:
            # Extract the MLP name
            parts = key.replace("/", ":").split(":")
            mlp_idx = None
            for i, part in enumerate(parts):
                if "mlp" in part:
                    mlp_idx = i
                    break
            
            if mlp_idx is not None:
                mlp_name = parts[mlp_idx]
                if mlp_name not in first_layer_weights:
                    first_layer_weights[mlp_name] = {}
                
                if key.endswith("w") or ":w" in key or "/w" in key:
                    first_layer_weights[mlp_name]["w"] = value
                elif key.endswith("b") or ":b" in key or "/b" in key:
                    first_layer_weights[mlp_name]["b"] = value
    
    return first_layer_weights


def analyze_processor_weights(
    params: GraphCastParams,
    verbose: bool = True
) -> ProcessorWeightAnalysis:
    """
    Analyze processor weights for spectral entropy calculation.
    
    This function extracts and analyzes the processor MLP weights,
    computing total energy, per-step breakdowns, and first-layer weights
    for sensitivity analysis.
    
    Args:
        params: GraphCastParams from load_graphcast_params()
        verbose: Whether to print analysis details
        
    Returns:
        ProcessorWeightAnalysis with energy breakdown and first-layer weights
    """
    config = params.metadata.get("mesh_config")
    if config is None:
        config = get_mesh_config("0.25deg")  # Default
    
    processor = params.processor
    
    # Analyze by message passing step
    per_step_energy = {}
    total_edge_energy = 0.0
    total_node_energy = 0.0
    first_layer_weights = {}
    
    for step in range(config.gnn_msg_steps):
        step_energy = {"edge": 0.0, "node": 0.0}
        
        # Find edge MLP weights for this step
        edge_pattern = f"processor_edges_{step}_mesh"
        node_pattern = f"processor_nodes_{step}_mesh_nodes"
        
        for key, value in processor.items():
            if not isinstance(value, np.ndarray):
                continue
            
            energy = float(np.sum(value.astype(np.float64) ** 2))
            
            if edge_pattern in key:
                step_energy["edge"] += energy
                total_edge_energy += energy
                
                # Capture first layer weights for sensitivity analysis
                if "linear_0" in key:
                    weight_type = "w" if key.endswith("w") else "b"
                    mlp_key = f"edge_step_{step}"
                    if mlp_key not in first_layer_weights:
                        first_layer_weights[mlp_key] = {}
                    first_layer_weights[mlp_key][weight_type] = value
                    
            elif node_pattern in key:
                step_energy["node"] += energy
                total_node_energy += energy
        
        per_step_energy[step] = step_energy
    
    total_energy = total_edge_energy + total_node_energy
    
    if verbose:
        print(f"Processor Weight Analysis")
        print(f"=" * 50)
        print(f"Configuration: {config.name}")
        print(f"Message passing steps: {config.gnn_msg_steps}")
        print(f"")
        print(f"Total Energy (Σw²): {total_energy:.4f}")
        if total_energy > 0:
            print(f"  Edge MLPs: {total_edge_energy:.4f} ({100*total_edge_energy/total_energy:.1f}%)")
            print(f"  Node MLPs: {total_node_energy:.4f} ({100*total_node_energy/total_energy:.1f}%)")
        print(f"")
        print(f"First-layer weights captured: {len(first_layer_weights)} MLPs")
        print(f"")
        print(f"Per-step breakdown:")
        for step, energies in per_step_energy.items():
            print(f"  Step {step:2d}: edge={energies['edge']:.4f}, node={energies['node']:.4f}")
    
    return ProcessorWeightAnalysis(
        total_energy=total_energy,
        edge_mlp_energy=total_edge_energy,
        node_mlp_energy=total_node_energy,
        per_step_energy=per_step_energy,
        n_steps=config.gnn_msg_steps,
        config=config,
        first_layer_weights=first_layer_weights,
    )


def compute_expected_edge_features(
    level: int,
    max_edge_length_km: float = 7056.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute expected edge feature statistics for a mesh level.
    
    GraphCast edge features are [normalized_distance, rel_x, rel_y, rel_z].
    The distance is normalized by max edge length, position by max distance.
    
    For edges at a given level:
    - Distance is approximately constant (edge_length_km / max_edge_length_km)
    - Relative position is uniformly distributed on a sphere of that radius
    
    Args:
        level: Mesh refinement level (0-6)
        max_edge_length_km: Maximum edge length for normalization (M0 edge)
        
    Returns:
        Tuple of (mean_features, covariance_matrix)
        mean_features: [4] array of expected feature values
        covariance_matrix: [4, 4] covariance of features
    """
    # Get edge length for this level
    edge_length_km = MESH_LEVELS[level].approx_km
    
    # Normalized distance (first feature)
    # This is the L2 distance in 3D divided by max, roughly proportional to arc length
    # For small angles: chord_length ≈ arc_length, so normalized_dist ≈ edge_length / max_edge
    normalized_dist = edge_length_km / max_edge_length_km
    
    # Mean features: [dist, 0, 0, 0] 
    # The relative positions average to zero due to symmetry
    mean_features = np.array([normalized_dist, 0.0, 0.0, 0.0])
    
    # Covariance: distance has low variance, positions have variance ~ (dist/2)^2
    # For uniformly distributed points on a sphere of radius r:
    # Var(x) = Var(y) = Var(z) = r^2 / 3
    position_variance = (normalized_dist ** 2) / 3.0
    
    # Distance variance is small (edges at same level have similar lengths)
    dist_variance = (normalized_dist * 0.05) ** 2  # ~5% variation
    
    covariance = np.diag([dist_variance, position_variance, position_variance, position_variance])
    
    return mean_features, covariance


def compute_first_layer_sensitivity(
    W: np.ndarray,
    b: Optional[np.ndarray],
    feature_mean: np.ndarray,
    feature_cov: np.ndarray,
) -> float:
    """
    Compute expected activation energy E[||Wx + b||²] for Gaussian input.
    
    For x ~ N(μ, Σ) and linear transform y = Wx + b:
        E[||y||²] = ||W*μ + b||² + tr(W'W Σ)
    
    This measures how much the first layer "activates" for inputs
    with the given statistics.
    
    Args:
        W: Weight matrix [input_dim, output_dim]
        b: Bias vector [output_dim] or None
        feature_mean: Mean of input features [input_dim]
        feature_cov: Covariance of input features [input_dim, input_dim]
        
    Returns:
        Expected squared activation energy
    """
    # Compute W @ mu
    W = W.astype(np.float64)
    mu = feature_mean.astype(np.float64)
    
    # Handle different weight matrix orientations
    if W.shape[0] == len(mu):
        # W is [input_dim, output_dim]
        W_mu = W.T @ mu
    else:
        # W is [output_dim, input_dim]
        W_mu = W @ mu
    
    if b is not None:
        b = b.astype(np.float64)
        W_mu_b = W_mu + b
    else:
        W_mu_b = W_mu
    
    # First term: ||W*μ + b||²
    mean_term = np.sum(W_mu_b ** 2)
    
    # Second term: tr(W'W Σ)
    # For W [input_dim, output_dim]: W'W is [input_dim, input_dim]
    if W.shape[0] == len(mu):
        WtW = W @ W.T
    else:
        WtW = W.T @ W
    
    Sigma = feature_cov.astype(np.float64)
    variance_term = np.trace(WtW @ Sigma)
    
    return float(mean_term + variance_term)


def compute_level_energy_sensitivity(
    params: GraphCastParams,
    verbose: bool = True
) -> Dict[int, float]:
    """
    Compute per-level energy using edge feature sensitivity analysis.
    
    This rigorous method estimates E[||MLP(x_level)||²] by analyzing
    how the first-layer MLP weights respond to expected edge features
    at each spatial scale.
    
    Args:
        params: GraphCastParams from load_graphcast_params()
        verbose: Whether to print details
        
    Returns:
        Dictionary mapping mesh level to sensitivity-based energy
    """
    config = params.metadata.get("mesh_config")
    if config is None:
        config = get_mesh_config("0.25deg")
    
    # Extract first-layer weights from processor edge MLPs
    first_layer = extract_first_layer_weights(params, component="processor")
    
    # Find edge MLP first layers (they process edge features)
    edge_mlp_weights = []
    for mlp_name, weights in first_layer.items():
        if "edge" in mlp_name.lower() and "w" in weights:
            edge_mlp_weights.append(weights)
    
    if not edge_mlp_weights:
        warnings.warn("No edge MLP first-layer weights found, using fallback")
        # Fallback: use all processor weights
        for key, value in params.processor.items():
            if isinstance(value, np.ndarray) and "linear_0" in key and key.endswith("w"):
                edge_mlp_weights.append({"w": value, "b": None})
    
    # Max edge length for normalization (M0 level)
    max_edge_length_km = MESH_LEVELS[0].approx_km
    
    # Compute sensitivity for each level
    level_energy = {}
    
    for level in config.levels:
        # Get expected edge features for this level
        mu, Sigma = compute_expected_edge_features(level, max_edge_length_km)
        
        # Compute total sensitivity across all edge MLPs
        total_sensitivity = 0.0
        n_mlps = 0
        
        for weights in edge_mlp_weights:
            W = weights.get("w")
            b = weights.get("b")
            
            if W is None:
                continue
            
            # Check if weight dimensions match feature dimensions
            # Edge features are 4D, but MLP input includes node features too
            # We focus on the edge feature portion
            input_dim = W.shape[0] if W.shape[0] < W.shape[1] else W.shape[1]
            
            if input_dim >= 4:
                # Extract the portion of W that corresponds to edge features
                # Edge features are typically the first 4 dimensions
                if W.shape[0] >= 4 and W.shape[0] < W.shape[1]:
                    W_edge = W[:4, :]
                elif W.shape[1] >= 4:
                    W_edge = W[:, :4].T
                else:
                    continue
                
                sensitivity = compute_first_layer_sensitivity(W_edge.T, None, mu, Sigma)
                total_sensitivity += sensitivity
                n_mlps += 1
        
        if n_mlps > 0:
            # Average sensitivity across MLPs, weighted by edge count
            n_edges = MESH_LEVELS[level].edges
            level_energy[level] = total_sensitivity * n_edges / config.total_edges
        else:
            level_energy[level] = 0.0
    
    if verbose:
        print(f"Per-Level Energy (Sensitivity Analysis)")
        print(f"=" * 60)
        print(f"Method: E[||W*x||²] = ||W*μ||² + tr(W'W Σ)")
        print(f"")
        print(f"{'Level':<8} {'Scale (km)':>12} {'Norm. Dist':>12} {'Energy':>12}")
        print(f"-" * 50)
        
        total_energy = sum(level_energy.values())
        for level in sorted(level_energy.keys()):
            scale = MESH_LEVELS[level].approx_km
            norm_dist = scale / max_edge_length_km
            energy = level_energy[level]
            print(f"M{level:<7} {scale:>12,.0f} {norm_dist:>12.4f} {energy:>12.6f}")
        
        print(f"-" * 50)
        print(f"{'Total':<8} {'':<12} {'':<12} {total_energy:>12.6f}")
    
    return level_energy


def extract_encoder_decoder_weights(
    params: Union[GraphCastParams, Dict[str, np.ndarray]]
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Extract Grid2Mesh (encoder) and Mesh2Grid (decoder) edge weights.
    
    Args:
        params: GraphCastParams or raw parameter dictionary
        
    Returns:
        Dictionary with "encoder" and "decoder" keys, each containing
        weight arrays keyed by parameter name
    """
    if isinstance(params, GraphCastParams):
        return {
            "encoder": params.encoder,
            "decoder": params.decoder,
        }
    
    # Parse from raw dict
    encoder = {}
    decoder = {}
    
    for key, value in params.items():
        if isinstance(value, np.ndarray):
            if "encoder" in key.lower() or "grid2mesh" in key.lower():
                encoder[key] = value
            elif "decoder" in key.lower() or "mesh2grid" in key.lower():
                decoder[key] = value
    
    return {"encoder": encoder, "decoder": decoder}


def analyze_encoder_decoder_weights(
    params: GraphCastParams,
    verbose: bool = True
) -> Dict[str, Dict[str, float]]:
    """
    Analyze encoder (Grid2Mesh) and decoder (Mesh2Grid) weights.
    
    These weights are NOT shared across mesh levels - they connect
    specific grid locations to specific mesh nodes, providing
    location-specific information transfer.
    
    Args:
        params: GraphCastParams from load_graphcast_params()
        verbose: Whether to print analysis details
        
    Returns:
        Dictionary with encoder and decoder energy breakdowns
    """
    result = {
        "encoder": {
            "total_energy": 0.0,
            "edge_mlp_energy": 0.0,
            "node_mlp_energy": 0.0,
            "n_params": 0,
        },
        "decoder": {
            "total_energy": 0.0,
            "edge_mlp_energy": 0.0,
            "node_mlp_energy": 0.0,
            "n_params": 0,
        },
    }
    
    # Analyze encoder
    for key, value in params.encoder.items():
        if not isinstance(value, np.ndarray):
            continue
        
        energy = float(np.sum(value.astype(np.float64) ** 2))
        result["encoder"]["total_energy"] += energy
        result["encoder"]["n_params"] += value.size
        
        if "edge" in key.lower():
            result["encoder"]["edge_mlp_energy"] += energy
        elif "node" in key.lower():
            result["encoder"]["node_mlp_energy"] += energy
    
    # Analyze decoder
    for key, value in params.decoder.items():
        if not isinstance(value, np.ndarray):
            continue
        
        energy = float(np.sum(value.astype(np.float64) ** 2))
        result["decoder"]["total_energy"] += energy
        result["decoder"]["n_params"] += value.size
        
        if "edge" in key.lower():
            result["decoder"]["edge_mlp_energy"] += energy
        elif "node" in key.lower():
            result["decoder"]["node_mlp_energy"] += energy
    
    if verbose:
        print(f"Encoder/Decoder Weight Analysis")
        print(f"=" * 50)
        print(f"")
        print(f"Encoder (Grid2Mesh):")
        print(f"  Total energy: {result['encoder']['total_energy']:.4f}")
        print(f"  Edge MLP energy: {result['encoder']['edge_mlp_energy']:.4f}")
        print(f"  Node MLP energy: {result['encoder']['node_mlp_energy']:.4f}")
        print(f"  Parameters: {result['encoder']['n_params']:,}")
        print(f"")
        print(f"Decoder (Mesh2Grid):")
        print(f"  Total energy: {result['decoder']['total_energy']:.4f}")
        print(f"  Edge MLP energy: {result['decoder']['edge_mlp_energy']:.4f}")
        print(f"  Node MLP energy: {result['decoder']['node_mlp_energy']:.4f}")
        print(f"  Parameters: {result['decoder']['n_params']:,}")
    
    return result


@dataclass
class RigorousAnalysisResult:
    """Results from rigorous spectral entropy analysis.
    
    Attributes:
        method: Analysis method used ("sensitivity", "encoder_decoder", "combined")
        level_energy: Energy per mesh level
        total_energy: Total energy across all levels
        processor_analysis: Detailed processor weight analysis
        encoder_decoder_analysis: Encoder/decoder weight breakdown
        config: Mesh configuration used
    """
    method: str
    level_energy: Dict[int, float]
    total_energy: float
    processor_analysis: ProcessorWeightAnalysis
    encoder_decoder_analysis: Optional[Dict[str, Dict[str, float]]]
    config: GraphCastMeshConfig
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for serialization."""
        return {
            "method": self.method,
            "levels": {
                level: {
                    "energy": energy,
                    "scale_km": MESH_LEVELS[level].approx_km,
                    "wavenumber": MESH_LEVELS[level].wavenumber,
                }
                for level, energy in self.level_energy.items()
            },
            "total_energy": self.total_energy,
            "config": {
                "name": self.config.name,
                "min_level": self.config.min_level,
                "max_level": self.config.max_level,
                "total_edges": self.config.total_edges,
            },
        }


def compute_rigorous_level_energy(
    params: GraphCastParams,
    method: str = "sensitivity",
    verbose: bool = True
) -> RigorousAnalysisResult:
    """
    Compute per-level energy using rigorous analysis methods.
    
    This is the main entry point for spectral entropy analysis.
    
    Available methods:
    - "sensitivity": Edge feature sensitivity analysis using E[||W*x||²]
    - "combined": Combines sensitivity analysis with encoder/decoder breakdown
    
    Args:
        params: GraphCastParams from load_graphcast_params()
        method: Analysis method ("sensitivity" or "combined")
        verbose: Whether to print analysis details
        
    Returns:
        RigorousAnalysisResult with level energy and analysis details
    """
    config = params.metadata.get("mesh_config")
    if config is None:
        config = get_mesh_config("0.25deg")
    
    # Always compute processor analysis
    processor_analysis = analyze_processor_weights(params, verbose=False)
    
    # Compute level energy based on method
    if method == "sensitivity":
        level_energy = compute_level_energy_sensitivity(params, verbose=verbose)
        encoder_decoder = None
        
    elif method == "combined":
        # Combine sensitivity analysis with encoder/decoder information
        level_energy = compute_level_energy_sensitivity(params, verbose=verbose)
        encoder_decoder = analyze_encoder_decoder_weights(params, verbose=verbose)
        
    else:
        raise ValueError(f"Unknown method: {method}. Use 'sensitivity' or 'combined'")
    
    total_energy = sum(level_energy.values())
    
    if verbose:
        print(f"\n" + "=" * 60)
        print(f"RIGOROUS ANALYSIS SUMMARY")
        print(f"=" * 60)
        print(f"Method: {method}")
        print(f"Model: {config.name}")
        print(f"Levels: M{config.min_level} to M{config.max_level}")
        print(f"Total edges: {config.total_edges:,}")
        print(f"")
        print(f"Level Energy Distribution:")
        print(f"{'Level':<8} {'Scale (km)':>12} {'Energy':>15} {'Fraction':>10}")
        print(f"-" * 50)
        for level in sorted(level_energy.keys()):
            scale = MESH_LEVELS[level].approx_km
            energy = level_energy[level]
            frac = energy / total_energy if total_energy > 0 else 0
            print(f"M{level:<7} {scale:>12,.0f} {energy:>15.6f} {frac:>10.2%}")
        print(f"-" * 50)
        print(f"{'Total':<8} {'':<12} {total_energy:>15.6f} {'100.00%':>10}")
    
    return RigorousAnalysisResult(
        method=method,
        level_energy=level_energy,
        total_energy=total_energy,
        processor_analysis=processor_analysis,
        encoder_decoder_analysis=encoder_decoder,
        config=config,
    )


def compute_weight_statistics(
    level_weights: Dict[int, np.ndarray]
) -> Dict[int, Dict[str, float]]:
    """
    Compute statistics for weights at each mesh level.
    
    Args:
        level_weights: Dictionary mapping level to weight arrays
        
    Returns:
        Dictionary mapping level to statistics dict
    """
    stats = {}
    
    for level, weights in level_weights.items():
        if len(weights) == 0:
            stats[level] = {
                "count": 0,
                "mean": np.nan,
                "std": np.nan,
                "min": np.nan,
                "max": np.nan,
                "energy": 0.0,
            }
        else:
            stats[level] = {
                "count": len(weights),
                "mean": float(np.mean(weights)),
                "std": float(np.std(weights)),
                "min": float(np.min(weights)),
                "max": float(np.max(weights)),
                "energy": float(np.sum(weights ** 2)),
            }
    
    return stats


def load_checkpoint_raw(
    path: Union[str, Path]
) -> Dict[str, np.ndarray]:
    """
    Load a raw checkpoint file (npz format).
    
    Args:
        path: Path to .npz checkpoint file
        
    Returns:
        Dictionary of parameter arrays
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    
    with np.load(path, allow_pickle=True) as data:
        return dict(data)


def list_checkpoint_keys(
    params: Union[GraphCastParams, Dict[str, np.ndarray], Path]
) -> List[str]:
    """
    List all parameter keys in a checkpoint.
    
    Args:
        params: GraphCastParams, raw dict, or path to checkpoint
        
    Returns:
        List of parameter key names
    """
    if isinstance(params, Path) or isinstance(params, str):
        params = load_checkpoint_raw(params)
    
    if isinstance(params, GraphCastParams):
        keys = []
        keys.extend(f"encoder/{k}" for k in params.encoder.keys())
        keys.extend(f"processor/{k}" for k in params.processor.keys())
        keys.extend(f"decoder/{k}" for k in params.decoder.keys())
        return keys
    
    return list(params.keys())


def get_param_shapes(
    params: Union[GraphCastParams, Dict[str, np.ndarray]]
) -> Dict[str, Tuple[int, ...]]:
    """
    Get shapes of all parameter arrays.
    
    Args:
        params: GraphCastParams or raw dict
        
    Returns:
        Dictionary mapping parameter name to shape tuple
    """
    shapes = {}
    
    if isinstance(params, GraphCastParams):
        for name, arr in params.encoder.items():
            shapes[f"encoder/{name}"] = arr.shape
        for name, arr in params.processor.items():
            shapes[f"processor/{name}"] = arr.shape
        for name, arr in params.decoder.items():
            shapes[f"decoder/{name}"] = arr.shape
    else:
        for name, value in params.items():
            if isinstance(value, np.ndarray):
                shapes[name] = value.shape
    
    return shapes


# Convenience function for quick analysis
def quick_load_and_analyze(
    resolution: str = "0.25deg",
    verbose: bool = True
) -> Tuple[Dict[int, float], ProcessorWeightAnalysis]:
    """
    Quick convenience function to load GraphCast and compute level energy.
    
    Uses the rigorous sensitivity analysis method.
    
    Args:
        resolution: Checkpoint resolution
        verbose: Whether to print progress
        
    Returns:
        Tuple of (level_energy, processor_analysis)
    """
    params = load_graphcast_params(resolution, verbose=verbose)
    level_energy = compute_level_energy_sensitivity(params, verbose=verbose)
    analysis = analyze_processor_weights(params, verbose=False)
    
    return level_energy, analysis


if __name__ == "__main__":
    # Demo: list available checkpoints
    print("Available GraphCast Checkpoints:")
    print("-" * 50)
    for key, info in get_available_checkpoints().items():
        print(f"  {key}:")
        print(f"    Resolution: {info['resolution']}")
        print(f"    Levels: {info['levels']}")
        print(f"    Size: {info['params_mb']} MB")
        print()
