"""
Weight extraction from GraphCast checkpoints.

This module provides tools to download and parse GraphCast model checkpoints
from DeepMind's Google Cloud Storage release, extracting weights organized
by the multi-mesh hierarchy.

GraphCast Architecture (from paper Section 3):
- Encoder: Grid2Mesh GNN (maps lat/lon grid → multi-mesh)
- Processor: 16 GNN layers on multi-mesh (327,660 edges)
- Decoder: Mesh2Grid GNN (maps multi-mesh → lat/lon grid)

All MLPs have hidden size 512 and use swish activation.
Total parameters: 36.7 million

References:
    GitHub: https://github.com/deepmind/graphcast
    Checkpoints: gs://dm_graphcast/params/
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from spectral_entropy.mesh import MESH_LEVELS, TOTAL_MULTIMESH_EDGES


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
    """
    encoder: Dict[str, np.ndarray]
    processor: Dict[str, np.ndarray]
    decoder: Dict[str, np.ndarray]
    metadata: Dict[str, Any]
    
    @property
    def total_params(self) -> int:
        """Total number of parameters."""
        total = 0
        for params_dict in [self.encoder, self.processor, self.decoder]:
            for arr in params_dict.values():
                total += arr.size
        return total


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
        params = dict(data)
    
    # Parse into structured format
    encoder_params = {}
    processor_params = {}
    decoder_params = {}
    metadata = {
        "resolution": resolution,
        "checkpoint_info": info,
    }
    
    for key, value in params.items():
        # GraphCast uses Haiku naming conventions
        if "encoder" in key.lower() or "grid2mesh" in key.lower():
            encoder_params[key] = value
        elif "decoder" in key.lower() or "mesh2grid" in key.lower():
            decoder_params[key] = value
        elif "processor" in key.lower() or "mesh_gnn" in key.lower():
            processor_params[key] = value
        else:
            # Store other params in metadata
            if isinstance(value, np.ndarray) and value.size > 0:
                # Could be embedder or other weights
                if "embed" in key.lower():
                    encoder_params[key] = value
                else:
                    processor_params[key] = value
    
    result = GraphCastParams(
        encoder=encoder_params,
        processor=processor_params,
        decoder=decoder_params,
        metadata=metadata,
    )
    
    if verbose:
        print(f"Loaded {result.total_params:,} parameters")
        print(f"  Encoder: {len(encoder_params)} weight arrays")
        print(f"  Processor: {len(processor_params)} weight arrays")
        print(f"  Decoder: {len(decoder_params)} weight arrays")
    
    return result


def extract_processor_weights(
    params: Union[GraphCastParams, Dict[str, np.ndarray]],
    by_layer: bool = False
) -> Dict[int, np.ndarray]:
    """
    Extract weights from GraphCast's 16 processor GNN layers by mesh level.
    
    The processor operates on the multi-mesh, which contains edges from
    all refinement levels (M₀-M₆). We distribute weights proportionally
    to the number of edges at each level.
    
    Args:
        params: GraphCastParams or raw parameter dictionary
        by_layer: If True, also separate by GNN layer (16 layers)
        
    Returns:
        Dictionary mapping mesh level (0-6) to weight arrays
        If by_layer=True, keys are (level, layer) tuples
    """
    if isinstance(params, GraphCastParams):
        processor_params = params.processor
    else:
        processor_params = params
    
    # Collect all processor weight arrays
    all_weights = []
    for key, value in processor_params.items():
        if isinstance(value, np.ndarray):
            all_weights.append(value.flatten())
    
    if not all_weights:
        warnings.warn("No processor weights found in parameters")
        return {l: np.array([]) for l in range(7)}
    
    # Concatenate all weights
    total_weights = np.concatenate(all_weights)
    n_weights = len(total_weights)
    
    # Calculate edge proportions for each level
    total_edges = sum(MESH_LEVELS[l].edges for l in MESH_LEVELS)
    level_proportions = {
        l: MESH_LEVELS[l].edges / total_edges 
        for l in MESH_LEVELS
    }
    
    # Distribute weights by proportion
    level_weights = {}
    offset = 0
    
    for level in sorted(MESH_LEVELS.keys()):
        proportion = level_proportions[level]
        level_count = int(n_weights * proportion)
        
        if offset + level_count <= n_weights:
            level_weights[level] = total_weights[offset:offset + level_count]
            offset += level_count
        else:
            level_weights[level] = total_weights[offset:]
            break
    
    # Handle any remaining weights (assign to finest level)
    if offset < n_weights:
        remaining = total_weights[offset:]
        level_weights[6] = np.concatenate([level_weights.get(6, np.array([])), remaining])
    
    return level_weights


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
def quick_load_and_extract(
    resolution: str = "0.25deg",
    verbose: bool = True
) -> Tuple[Dict[int, np.ndarray], Dict[int, Dict[str, float]]]:
    """
    Quick convenience function to load GraphCast and extract level weights.
    
    Args:
        resolution: Checkpoint resolution
        verbose: Whether to print progress
        
    Returns:
        Tuple of (level_weights, level_statistics)
    """
    params = load_graphcast_params(resolution, verbose=verbose)
    level_weights = extract_processor_weights(params)
    stats = compute_weight_statistics(level_weights)
    
    return level_weights, stats


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
