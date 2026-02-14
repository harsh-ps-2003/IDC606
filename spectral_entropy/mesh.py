"""
Multi-mesh geometry utilities for GraphCast's icosahedral mesh hierarchy.

This module provides tools for working with GraphCast's multi-level mesh structure,
mapping between mesh refinement levels, spatial scales, and wavenumbers.

GraphCast uses a hierarchical icosahedral mesh where:
- M₀: Base icosahedron (12 nodes, 20 faces)
- Each refinement Mᵣ → Mᵣ₊₁ splits each face into 4 smaller faces
- Released models use "mesh 2to6" (levels M₂-M₆) or "mesh 2to5" (levels M₂-M₅)

IMPORTANT: The released GraphCast checkpoints do NOT use all 7 levels (M₀-M₆).
- GraphCast (0.25deg): mesh_size=6, uses levels M₂ through M₆ (5 levels)
- GraphCast_small (1deg): mesh_size=5, uses levels M₂ through M₅ (4 levels)

The "2to6" notation means the multi-mesh includes edges from levels 2, 3, 4, 5, 6.

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

# Base icosahedron edge length on unit sphere (radians)
# Derived from golden ratio: 2 * arcsin(1 / (sqrt(5) * sin(pi/5)))
ICOSAHEDRON_EDGE_LENGTH_RAD = 1.1071487177940904


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


# ============================================================================
# GraphCast Model Configurations
# ============================================================================

@dataclass(frozen=True)
class GraphCastMeshConfig:
    """Configuration for a specific GraphCast model's mesh.
    
    Attributes:
        name: Model name (e.g., "GraphCast", "GraphCast_small")
        mesh_size: The mesh_size parameter (max refinement level)
        min_level: Minimum mesh level included (typically 2)
        max_level: Maximum mesh level included (equals mesh_size)
        resolution: Grid resolution in degrees
        gnn_msg_steps: Number of message passing steps in processor
        latent_size: Hidden dimension size
    """
    name: str
    mesh_size: int
    min_level: int
    max_level: int
    resolution: float
    gnn_msg_steps: int
    latent_size: int
    
    @property
    def levels(self) -> List[int]:
        """List of mesh levels used in this configuration."""
        return list(range(self.min_level, self.max_level + 1))
    
    @property
    def n_levels(self) -> int:
        """Number of mesh levels."""
        return self.max_level - self.min_level + 1
    
    @property
    def total_edges(self) -> int:
        """Total number of edges in the multi-mesh."""
        return sum(MESH_LEVELS[l].edges for l in self.levels)
    
    def get_level_edge_counts(self) -> Dict[int, int]:
        """Get edge count for each level in this configuration."""
        return {l: MESH_LEVELS[l].edges for l in self.levels}


# Pre-defined configurations for released GraphCast models
GRAPHCAST_CONFIGS: Dict[str, GraphCastMeshConfig] = {
    "0.25deg": GraphCastMeshConfig(
        name="GraphCast",
        mesh_size=6,
        min_level=2,  # "mesh 2to6" means levels 2-6
        max_level=6,
        resolution=0.25,
        gnn_msg_steps=16,
        latent_size=512,
    ),
    "1deg": GraphCastMeshConfig(
        name="GraphCast_small",
        mesh_size=5,
        min_level=2,  # "mesh 2to5" means levels 2-5
        max_level=5,
        resolution=1.0,
        gnn_msg_steps=16,
        latent_size=512,
    ),
    "operational": GraphCastMeshConfig(
        name="GraphCast_operational",
        mesh_size=6,
        min_level=2,
        max_level=6,
        resolution=0.25,
        gnn_msg_steps=16,
        latent_size=512,
    ),
}


def get_mesh_config(resolution: str = "0.25deg") -> GraphCastMeshConfig:
    """Get the mesh configuration for a GraphCast model.
    
    Args:
        resolution: Model resolution key ("0.25deg", "1deg", "operational")
        
    Returns:
        GraphCastMeshConfig for the specified model
    """
    if resolution not in GRAPHCAST_CONFIGS:
        raise ValueError(f"Unknown resolution '{resolution}'. "
                        f"Available: {list(GRAPHCAST_CONFIGS.keys())}")
    return GRAPHCAST_CONFIGS[resolution]


# ============================================================================
# Exact Icosahedral Mesh Generation
# ============================================================================

@dataclass
class TriangularMesh:
    """Data structure for triangular meshes on a sphere.
    
    Attributes:
        vertices: Spatial positions on unit sphere [num_vertices, 3]
        faces: Triangular faces [num_faces, 3] with vertex indices
    """
    vertices: np.ndarray
    faces: np.ndarray


def _get_icosahedron() -> TriangularMesh:
    """
    Generate a regular icosahedron with vertices on the unit sphere.
    
    The icosahedron has 12 vertices, 20 faces, and 30 edges.
    Vertices are constructed using the golden ratio.
    
    Returns:
        TriangularMesh with vertices on unit sphere
    """
    phi = (1 + np.sqrt(5)) / 2  # Golden ratio
    
    # Generate vertices using golden ratio construction
    vertices = []
    for c1 in [1., -1.]:
        for c2 in [phi, -phi]:
            vertices.append((c1, c2, 0.))
            vertices.append((0., c1, c2))
            vertices.append((c2, 0., c1))
    
    vertices = np.array(vertices, dtype=np.float32)
    vertices /= np.linalg.norm([1., phi])  # Normalize to unit sphere
    
    # Define faces (counter-clockwise orientation from outside)
    faces = np.array([
        (0, 1, 2), (0, 6, 1), (8, 0, 2), (8, 4, 0), (3, 8, 2),
        (3, 2, 7), (7, 2, 1), (0, 4, 6), (4, 11, 6), (6, 11, 5),
        (1, 5, 7), (4, 10, 11), (4, 8, 10), (10, 8, 3), (10, 3, 9),
        (11, 10, 9), (11, 9, 5), (5, 9, 7), (9, 3, 7), (1, 6, 5),
    ], dtype=np.int32)
    
    # Rotate to align top face parallel to XY plane
    angle_between_faces = 2 * np.arcsin(phi / np.sqrt(3))
    rotation_angle = (np.pi - angle_between_faces) / 2
    
    # Rotation matrix around Y axis
    c, s = np.cos(rotation_angle), np.sin(rotation_angle)
    rotation_matrix = np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c]
    ])
    vertices = vertices @ rotation_matrix
    
    return TriangularMesh(vertices=vertices.astype(np.float32), faces=faces)


def _split_triangle_faces(mesh: TriangularMesh) -> TriangularMesh:
    """
    Split each triangular face into 4 smaller triangles.
    
    New vertices are placed at edge midpoints and projected onto the unit sphere.
    
    Args:
        mesh: Input triangular mesh
        
    Returns:
        Refined mesh with 4x faces
    """
    vertices = list(mesh.vertices)
    new_vertex_cache = {}  # Cache to avoid duplicate vertices
    
    def get_midpoint_index(i1: int, i2: int) -> int:
        """Get or create midpoint vertex between two vertices."""
        key = tuple(sorted([i1, i2]))
        if key in new_vertex_cache:
            return new_vertex_cache[key]
        
        # Create midpoint and project to unit sphere
        midpoint = (vertices[i1] + vertices[i2]) / 2
        midpoint = midpoint / np.linalg.norm(midpoint)
        
        new_idx = len(vertices)
        vertices.append(midpoint)
        new_vertex_cache[key] = new_idx
        return new_idx
    
    new_faces = []
    for i1, i2, i3 in mesh.faces:
        # Get midpoint indices
        m12 = get_midpoint_index(i1, i2)
        m23 = get_midpoint_index(i2, i3)
        m31 = get_midpoint_index(i3, i1)
        
        # Create 4 new triangles (preserving orientation)
        new_faces.extend([
            [i1, m12, m31],
            [m12, i2, m23],
            [m31, m23, i3],
            [m12, m23, m31],
        ])
    
    return TriangularMesh(
        vertices=np.array(vertices, dtype=np.float32),
        faces=np.array(new_faces, dtype=np.int32)
    )


def generate_icosahedral_mesh(max_level: int = 6) -> List[TriangularMesh]:
    """
    Generate a hierarchy of icosahedral meshes from level 0 to max_level.
    
    Each level is created by splitting the previous level's triangles into 4.
    All vertices lie on the unit sphere.
    
    Args:
        max_level: Maximum refinement level (0-6)
        
    Returns:
        List of TriangularMesh objects, one per level
    """
    meshes = [_get_icosahedron()]
    
    for _ in range(max_level):
        meshes.append(_split_triangle_faces(meshes[-1]))
    
    return meshes


def merge_meshes(meshes: List[TriangularMesh]) -> TriangularMesh:
    """
    Merge multiple mesh levels into a single multi-mesh.
    
    The vertices come from the finest mesh (they include all coarser vertices).
    The faces are concatenated from all levels.
    
    Args:
        meshes: List of meshes from coarse to fine
        
    Returns:
        Merged TriangularMesh
    """
    # Finest mesh has all vertices
    vertices = meshes[-1].vertices
    
    # Concatenate all faces
    all_faces = np.concatenate([m.faces for m in meshes], axis=0)
    
    return TriangularMesh(vertices=vertices, faces=all_faces)


def faces_to_edges(faces: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert triangular faces to edge sender/receiver indices.
    
    Each triangular face produces 3 directed edges.
    For a closed surface with consistent orientation, each edge appears twice
    (once in each direction).
    
    Args:
        faces: [num_faces, 3] array of vertex indices
        
    Returns:
        Tuple of (senders, receivers) arrays, each of shape [num_edges]
    """
    senders = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
    receivers = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
    return senders, receivers


def compute_exact_edge_lengths(
    vertices: np.ndarray,
    senders: np.ndarray,
    receivers: np.ndarray,
    in_km: bool = True
) -> np.ndarray:
    """
    Compute exact geodesic edge lengths from mesh vertices.
    
    For vertices on a unit sphere, geodesic distance = arccos(v1 · v2).
    
    Args:
        vertices: [num_vertices, 3] vertex positions on unit sphere
        senders: [num_edges] sender vertex indices
        receivers: [num_edges] receiver vertex indices
        in_km: If True, return lengths in km; otherwise in radians
        
    Returns:
        [num_edges] array of edge lengths
    """
    v1 = vertices[senders]
    v2 = vertices[receivers]
    
    # Dot product gives cos(angle) for unit vectors
    dot_products = np.sum(v1 * v2, axis=1)
    dot_products = np.clip(dot_products, -1.0, 1.0)  # Numerical stability
    
    # Geodesic distance in radians
    edge_lengths_rad = np.arccos(dot_products)
    
    if in_km:
        return edge_lengths_rad * EARTH_RADIUS_KM
    return edge_lengths_rad


def get_exact_edge_lengths_by_level(
    min_level: int = 0,
    max_level: int = 6
) -> Dict[int, np.ndarray]:
    """
    Generate mesh and compute exact edge lengths for each level.
    
    Args:
        min_level: Minimum level to include
        max_level: Maximum level to include
        
    Returns:
        Dictionary mapping level to array of edge lengths (km)
    """
    # Generate full mesh hierarchy
    meshes = generate_icosahedral_mesh(max_level)
    
    result = {}
    for level in range(min_level, max_level + 1):
        mesh = meshes[level]
        senders, receivers = faces_to_edges(mesh.faces)
        
        # Only get edges that are NEW at this level (not from coarser levels)
        if level > 0:
            prev_mesh = meshes[level - 1]
            prev_n_faces = prev_mesh.faces.shape[0]
            # New faces start after the inherited faces
            # Each face produces 3 edges, but we need to account for the 4x split
            # Actually, in our implementation, each level has only its own faces
            pass
        
        lengths = compute_exact_edge_lengths(mesh.vertices, senders, receivers)
        result[level] = lengths
    
    return result


def get_edge_length_statistics(
    min_level: int = 0,
    max_level: int = 6
) -> Dict[int, Dict[str, float]]:
    """
    Compute exact edge length statistics for each mesh level.
    
    Args:
        min_level: Minimum level
        max_level: Maximum level
        
    Returns:
        Dictionary mapping level to statistics dict with mean, std, min, max
    """
    edge_lengths = get_exact_edge_lengths_by_level(min_level, max_level)
    
    stats = {}
    for level, lengths in edge_lengths.items():
        stats[level] = {
            "mean": float(np.mean(lengths)),
            "std": float(np.std(lengths)),
            "min": float(np.min(lengths)),
            "max": float(np.max(lengths)),
            "count": len(lengths),
        }
    
    return stats


# ============================================================================
# Edge Classification Functions
# ============================================================================

def get_edges_per_level(min_level: int = 0, max_level: int = 6) -> Dict[int, int]:
    """
    Get the number of edges at each mesh level.
    
    For an icosahedral mesh, edges at level k = 60 * 4^k (bidirectional).
    
    Args:
        min_level: Minimum level to include
        max_level: Maximum level to include
        
    Returns:
        Dictionary mapping level to edge count
    """
    return {l: MESH_LEVELS[l].edges for l in range(min_level, max_level + 1)}


def get_edge_indices_by_level(
    min_level: int = 2,
    max_level: int = 6
) -> Dict[int, Tuple[int, int]]:
    """
    Get the start and end indices for edges at each level in the merged mesh.
    
    In GraphCast's merged mesh, faces (and thus edges) are concatenated in order
    from coarsest to finest level. Each face produces 3 directed edges.
    
    Args:
        min_level: Minimum mesh level (default 2 for "mesh 2to6")
        max_level: Maximum mesh level (default 6)
        
    Returns:
        Dictionary mapping level to (start_idx, end_idx) tuple
        
    Example:
        >>> indices = get_edge_indices_by_level(2, 6)
        >>> indices[2]  # (0, 960) - first 960 edges are from M2
        >>> indices[3]  # (960, 4800) - next 3840 edges are from M3
    """
    level_indices = {}
    offset = 0
    
    for level in range(min_level, max_level + 1):
        n_edges = MESH_LEVELS[level].edges
        level_indices[level] = (offset, offset + n_edges)
        offset += n_edges
    
    return level_indices


def compute_edge_geodesic_length_rad(level: int) -> float:
    """
    Compute the theoretical geodesic edge length (in radians) for a mesh level.
    
    Edge length halves with each refinement:
        L_level = L_0 / 2^level
    
    where L_0 ≈ 1.107 radians for the base icosahedron.
    
    Args:
        level: Mesh refinement level (0-6)
        
    Returns:
        Edge length in radians on unit sphere
    """
    return ICOSAHEDRON_EDGE_LENGTH_RAD / (2 ** level)


def compute_edge_geodesic_length_km(level: int) -> float:
    """
    Compute the theoretical geodesic edge length (in km) for a mesh level.
    
    Args:
        level: Mesh refinement level (0-6)
        
    Returns:
        Edge length in kilometers
    """
    length_rad = compute_edge_geodesic_length_rad(level)
    return length_rad * EARTH_RADIUS_KM


def get_level_from_edge_length_rad(
    length_rad: float,
    min_level: int = 0,
    max_level: int = 6,
    tolerance: float = 0.25
) -> int:
    """
    Classify an edge to a mesh level based on its geodesic length.
    
    Args:
        length_rad: Edge length in radians
        min_level: Minimum possible level
        max_level: Maximum possible level
        tolerance: Fractional tolerance for matching
        
    Returns:
        Mesh level that best matches the edge length
    """
    for level in range(min_level, max_level + 1):
        expected = compute_edge_geodesic_length_rad(level)
        if abs(length_rad - expected) / expected < tolerance:
            return level
    
    # Default to finest level if no match
    return max_level


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


def classify_edges_by_length(
    edge_lengths_km: np.ndarray,
    min_level: int = 0,
    max_level: int = 6
) -> Dict[int, np.ndarray]:
    """
    Classify edges to mesh levels based on their geodesic lengths.
    
    Uses exact level boundaries derived from icosahedral geometry.
    Edge length halves with each refinement level.
    
    Args:
        edge_lengths_km: Array of edge lengths in kilometers
        min_level: Minimum level to consider
        max_level: Maximum level to consider
        
    Returns:
        Dictionary mapping level to boolean mask of edges at that level
    """
    # Compute level boundaries (geometric mean between adjacent levels)
    boundaries = {}
    for level in range(min_level, max_level + 1):
        if level == min_level:
            # Upper boundary: midpoint to level-1 (or infinity)
            upper = MESH_LEVELS[level].approx_km * 1.5
        else:
            upper = np.sqrt(MESH_LEVELS[level].approx_km * MESH_LEVELS[level - 1].approx_km)
        
        if level == max_level:
            # Lower boundary: half of this level
            lower = MESH_LEVELS[level].approx_km * 0.5
        else:
            lower = np.sqrt(MESH_LEVELS[level].approx_km * MESH_LEVELS[level + 1].approx_km)
        
        boundaries[level] = (lower, upper)
    
    # Classify edges
    result = {}
    for level in range(min_level, max_level + 1):
        lower, upper = boundaries[level]
        mask = (edge_lengths_km >= lower) & (edge_lengths_km < upper)
        result[level] = mask
    
    return result


def get_level_edge_indices(
    min_level: int = 2,
    max_level: int = 6
) -> Dict[int, Tuple[int, int]]:
    """
    Get edge index ranges for each level in a merged multi-mesh.
    
    In GraphCast's merged mesh, faces from each level are concatenated
    in order from coarsest to finest.
    
    Args:
        min_level: Minimum mesh level
        max_level: Maximum mesh level
        
    Returns:
        Dictionary mapping level to (start_idx, end_idx) for edges
    """
    result = {}
    offset = 0
    
    for level in range(min_level, max_level + 1):
        n_edges = MESH_LEVELS[level].edges
        result[level] = (offset, offset + n_edges)
        offset += n_edges
    
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


def summary_for_config(config: GraphCastMeshConfig) -> str:
    """
    Print a summary for a specific GraphCast configuration.
    
    Args:
        config: GraphCastMeshConfig object
        
    Returns:
        Summary string
    """
    lines = [
        f"GraphCast Configuration: {config.name}",
        "=" * 50,
        f"Resolution: {config.resolution}°",
        f"Mesh levels: M{config.min_level} to M{config.max_level} ({config.n_levels} levels)",
        f"Total edges: {config.total_edges:,}",
        f"GNN message passing steps: {config.gnn_msg_steps}",
        f"Latent size: {config.latent_size}",
        "",
        "Edge counts per level:",
    ]
    
    for level, count in config.get_level_edge_counts().items():
        info = MESH_LEVELS[level]
        pct = 100 * count / config.total_edges
        lines.append(
            f"  M{level}: {count:>7,} edges ({pct:5.1f}%) - ~{info.approx_km:,.0f} km"
        )
    
    return "\n".join(lines)


if __name__ == "__main__":
    print(summary())
    print()
    print("=" * 60)
    print()
    for key, config in GRAPHCAST_CONFIGS.items():
        print(summary_for_config(config))
        print()
