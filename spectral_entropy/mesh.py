"""
Multi-mesh geometry utilities for GraphCast's icosahedral mesh hierarchy.

This module provides tools for working with GraphCast's multi-level mesh structure,
mapping between mesh refinement levels, spatial scales, and wavenumbers.

GraphCast uses a hierarchical icosahedral mesh where:
- M₀: Base icosahedron (12 nodes, 20 faces)
- Each refinement Mᵣ → Mᵣ₊₁ splits each face into 4 smaller faces
- Multi-mesh contains nodes from M₆ and edges from ALL levels (M₀-M₆)

References:
    GraphCast paper: https://arxiv.org/pdf/2212.12794
    Table 4 in supplementary materials contains mesh statistics
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

# Earth's circumference at the equator in kilometers
EARTH_CIRCUMFERENCE_KM = 40_075.0

# Earth's radius in kilometers
EARTH_RADIUS_KM = 6_371.0


@dataclass(frozen=True)
class MeshLevelInfo:
    """Information about a single mesh refinement level.
    
    Attributes:
        level: Refinement level (0 = base icosahedron)
        nodes: Number of vertices in the mesh
        faces: Number of triangular faces
        edges: Number of edges (bidirectional, counted twice)
        multilevel_edges: Cumulative edges including all coarser levels
        approx_km: Approximate spatial scale in kilometers
    """
    level: int
    nodes: int
    faces: int
    edges: int
    multilevel_edges: int
    approx_km: float
    
    @property
    def wavenumber(self) -> float:
        """Wavenumber k = 1/L in 1/km."""
        return 1.0 / self.approx_km
    
    @property
    def angular_wavenumber(self) -> float:
        """Angular wavenumber for spherical harmonics ℓ ≈ 2πR/L."""
        return 2 * np.pi * EARTH_RADIUS_KM / self.approx_km


# GraphCast mesh statistics from paper Table 4
# Edge counts are bidirectional (each edge counted twice)
MESH_LEVELS: Dict[int, MeshLevelInfo] = {
    0: MeshLevelInfo(
        level=0,
        nodes=12,
        faces=20,
        edges=60,
        multilevel_edges=60,
        approx_km=7000.0  # Planetary scale
    ),
    1: MeshLevelInfo(
        level=1,
        nodes=42,
        faces=80,
        edges=240,
        multilevel_edges=300,
        approx_km=3500.0  # Synoptic scale
    ),
    2: MeshLevelInfo(
        level=2,
        nodes=162,
        faces=320,
        edges=960,
        multilevel_edges=1260,
        approx_km=1700.0  # Large-scale
    ),
    3: MeshLevelInfo(
        level=3,
        nodes=642,
        faces=1280,
        edges=3840,
        multilevel_edges=5100,
        approx_km=850.0  # Frontal scale
    ),
    4: MeshLevelInfo(
        level=4,
        nodes=2562,
        faces=5120,
        edges=15360,
        multilevel_edges=20460,
        approx_km=425.0  # Regional scale
    ),
    5: MeshLevelInfo(
        level=5,
        nodes=10242,
        faces=20480,
        edges=61440,
        multilevel_edges=81900,
        approx_km=212.0  # Mesoscale
    ),
    6: MeshLevelInfo(
        level=6,
        nodes=40962,
        faces=81920,
        edges=245760,
        multilevel_edges=327660,
        approx_km=100.0  # Dissipative scale
    ),
}

# Total multi-mesh edges (union of all levels)
TOTAL_MULTIMESH_EDGES = 327660

# Physical analogs for each mesh level (for documentation/interpretation)
PHYSICAL_ANALOGS: Dict[int, str] = {
    0: "Planetary waves / Large-scale forcing",
    1: "Synoptic background flow",
    2: "Large-scale vortices (Highs/Lows)",
    3: "Frontal systems / Inertial range start",
    4: "Regional moisture transport",
    5: "Mesoscale convective systems",
    6: "Localized turbulence / Dissipative range",
}


def compute_wavenumber(level: int) -> float:
    """
    Compute wavenumber k = 1/L for a given mesh level.
    
    Args:
        level: Mesh refinement level (0-6)
        
    Returns:
        Wavenumber in 1/km
        
    Raises:
        ValueError: If level is not in valid range [0, 6]
    """
    if level not in MESH_LEVELS:
        raise ValueError(f"Level must be in range [0, 6], got {level}")
    return MESH_LEVELS[level].wavenumber


def get_level_spatial_scale(level: int) -> float:
    """
    Get the approximate spatial scale (in km) for a mesh level.
    
    The spatial scale is derived from the edge length of the icosahedral
    mesh at each refinement level:
        L_r ≈ C_earth / (5 × 2^r)
    
    where C_earth is Earth's circumference (~40,075 km).
    
    Args:
        level: Mesh refinement level (0-6)
        
    Returns:
        Spatial scale in kilometers
    """
    if level not in MESH_LEVELS:
        raise ValueError(f"Level must be in range [0, 6], got {level}")
    return MESH_LEVELS[level].approx_km


def compute_theoretical_edge_length(level: int) -> float:
    """
    Compute theoretical edge length for icosahedral mesh at given level.
    
    For a refined icosahedron projected onto a sphere:
        L_r ≈ C_earth / (5 × 2^r)
    
    This is an approximation; actual edge lengths vary slightly
    across the sphere due to projection effects.
    
    Args:
        level: Mesh refinement level (0-6)
        
    Returns:
        Theoretical edge length in kilometers
    """
    return EARTH_CIRCUMFERENCE_KM / (5 * (2 ** level))


def get_edge_length_distribution(level: int) -> Dict[str, float]:
    """
    Get statistics about edge lengths at a given mesh level.
    
    Since icosahedral meshes have nearly uniform edge lengths,
    we return theoretical values with small variations.
    
    Args:
        level: Mesh refinement level (0-6)
        
    Returns:
        Dictionary with 'mean', 'min', 'max', 'std' edge lengths in km
    """
    mean_length = compute_theoretical_edge_length(level)
    # Icosahedral meshes have ~5% variation in edge lengths
    variation = 0.05 * mean_length
    
    return {
        "mean": mean_length,
        "min": mean_length - variation,
        "max": mean_length + variation,
        "std": variation / 2,
    }


def get_all_wavenumbers() -> Tuple[np.ndarray, np.ndarray]:
    """
    Get arrays of mesh levels and corresponding wavenumbers.
    
    Returns:
        Tuple of (levels, wavenumbers) as numpy arrays
    """
    levels = np.array(list(MESH_LEVELS.keys()))
    wavenumbers = np.array([MESH_LEVELS[l].wavenumber for l in levels])
    return levels, wavenumbers


def get_all_spatial_scales() -> Tuple[np.ndarray, np.ndarray]:
    """
    Get arrays of mesh levels and corresponding spatial scales.
    
    Returns:
        Tuple of (levels, scales_km) as numpy arrays
    """
    levels = np.array(list(MESH_LEVELS.keys()))
    scales = np.array([MESH_LEVELS[l].approx_km for l in levels])
    return levels, scales


def classify_edge_by_length(
    edge_length_km: float,
    tolerance: float = 0.3
) -> int:
    """
    Classify an edge to a mesh level based on its length.
    
    GraphCast's multi-mesh contains edges from ALL refinement levels.
    This function determines which level an edge belongs to based on
    its geodesic length.
    
    Args:
        edge_length_km: Length of the edge in kilometers
        tolerance: Fractional tolerance for matching (default 0.3 = 30%)
        
    Returns:
        Mesh level (0-6) that best matches the edge length
    """
    # Sort levels by spatial scale (descending)
    for level in sorted(MESH_LEVELS.keys()):
        expected_length = MESH_LEVELS[level].approx_km
        lower_bound = expected_length * (1 - tolerance)
        upper_bound = expected_length * (1 + tolerance)
        
        if lower_bound <= edge_length_km <= upper_bound:
            return level
    
    # If no match, assign to finest level
    return 6


def geodesic_distance(
    lat1: float, lon1: float,
    lat2: float, lon2: float
) -> float:
    """
    Compute geodesic (great-circle) distance between two points.
    
    Uses the Haversine formula for numerical stability.
    
    Args:
        lat1, lon1: Latitude and longitude of first point (degrees)
        lat2, lon2: Latitude and longitude of second point (degrees)
        
    Returns:
        Distance in kilometers
    """
    # Convert to radians
    lat1_r, lon1_r = np.radians(lat1), np.radians(lon1)
    lat2_r, lon2_r = np.radians(lat2), np.radians(lon2)
    
    # Haversine formula
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return EARTH_RADIUS_KM * c


def map_weights_to_levels(
    weight_dict: Dict[str, np.ndarray],
    edge_positions: Optional[np.ndarray] = None
) -> Dict[int, np.ndarray]:
    """
    Map neural network weights to mesh levels based on edge positions.
    
    This function assigns weights from GraphCast's processor to their
    corresponding mesh levels. If edge positions are provided, classification
    is done by geodesic distance. Otherwise, we use heuristics based on
    the weight tensor shapes.
    
    Args:
        weight_dict: Dictionary of weight arrays from GraphCast checkpoint
        edge_positions: Optional array of edge endpoint positions
            Shape: (n_edges, 2, 3) for 3D positions or (n_edges, 2, 2) for lat/lon
            
    Returns:
        Dictionary mapping mesh level (0-6) to concatenated weight arrays
    """
    level_weights: Dict[int, List[np.ndarray]] = {l: [] for l in MESH_LEVELS}
    
    if edge_positions is not None:
        # Classify by actual edge lengths
        n_edges = edge_positions.shape[0]
        
        for i in range(n_edges):
            p1, p2 = edge_positions[i]
            
            # Handle 3D Cartesian or lat/lon coordinates
            if len(p1) == 3:
                # Convert 3D to lat/lon
                x, y, z = p1
                lat1 = np.degrees(np.arcsin(z / np.sqrt(x**2 + y**2 + z**2)))
                lon1 = np.degrees(np.arctan2(y, x))
                x, y, z = p2
                lat2 = np.degrees(np.arcsin(z / np.sqrt(x**2 + y**2 + z**2)))
                lon2 = np.degrees(np.arctan2(y, x))
            else:
                lat1, lon1 = p1
                lat2, lon2 = p2
            
            distance = geodesic_distance(lat1, lon1, lat2, lon2)
            level = classify_edge_by_length(distance)
            
            # Assuming weights are indexed by edge
            if "edge_weights" in weight_dict:
                level_weights[level].append(weight_dict["edge_weights"][i])
    else:
        # Use heuristics based on number of edges at each level
        # This is less accurate but works without position data
        total_edges = sum(MESH_LEVELS[l].edges for l in MESH_LEVELS)
        
        for key, weights in weight_dict.items():
            if weights.ndim >= 1:
                n = weights.shape[0]
                
                # Distribute weights proportionally to edge counts
                offset = 0
                for level in sorted(MESH_LEVELS.keys()):
                    level_edges = MESH_LEVELS[level].edges
                    fraction = level_edges / total_edges
                    level_count = int(n * fraction)
                    
                    if level_count > 0 and offset + level_count <= n:
                        level_weights[level].append(
                            weights[offset:offset + level_count].flatten()
                        )
                        offset += level_count
    
    # Concatenate weights for each level
    result = {}
    for level, weight_list in level_weights.items():
        if weight_list:
            result[level] = np.concatenate([w.flatten() for w in weight_list])
        else:
            result[level] = np.array([])
    
    return result


def get_level_info_table() -> str:
    """
    Generate a formatted table of mesh level information.
    
    Returns:
        Markdown-formatted table string
    """
    header = "| Level | Nodes | Edges | Scale (km) | Wavenumber (1/km) | Physical Analog |"
    separator = "|-------|-------|-------|------------|-------------------|-----------------|"
    
    rows = [header, separator]
    for level, info in MESH_LEVELS.items():
        analog = PHYSICAL_ANALOGS.get(level, "N/A")
        rows.append(
            f"| M{level} | {info.nodes:,} | {info.edges:,} | "
            f"~{info.approx_km:,.0f} | {info.wavenumber:.2e} | {analog} |"
        )
    
    return "\n".join(rows)


def summary() -> str:
    """
    Print a summary of the mesh hierarchy.
    
    Returns:
        Summary string
    """
    lines = [
        "GraphCast Multi-Mesh Hierarchy",
        "=" * 40,
        f"Earth circumference: {EARTH_CIRCUMFERENCE_KM:,.0f} km",
        f"Total multi-mesh edges: {TOTAL_MULTIMESH_EDGES:,}",
        f"Refinement levels: 0-6 (7 total)",
        "",
        get_level_info_table(),
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    print(summary())
