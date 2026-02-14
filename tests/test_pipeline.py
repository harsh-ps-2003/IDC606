#!/usr/bin/env python
"""
Test script for the spectral entropy analysis pipeline.

This script tests all components of the pipeline:
1. Mesh utilities
2. Weight energy computation
3. Spectral entropy calculation
4. Power law fitting
5. Visualization (optional)

Run with: python -m tests.test_pipeline
"""

import sys
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_mesh_module():
    """Test mesh.py functionality."""
    print("\n" + "=" * 60)
    print("Testing: mesh.py")
    print("=" * 60)
    
    from spectral_entropy.mesh import (
        MESH_LEVELS,
        EARTH_CIRCUMFERENCE_KM,
        compute_wavenumber,
        get_level_spatial_scale,
        get_edge_length_distribution,
        classify_edge_by_length,
        geodesic_distance,
    )
    
    # Test constants
    assert EARTH_CIRCUMFERENCE_KM == 40_075.0
    assert len(MESH_LEVELS) == 7
    print(f"✓ Constants loaded correctly")
    
    # Test wavenumber computation
    k0 = compute_wavenumber(0)
    k6 = compute_wavenumber(6)
    assert k0 < k6  # Finer levels have higher wavenumbers
    print(f"✓ Wavenumber: M0 = {k0:.2e}, M6 = {k6:.2e}")
    
    # Test spatial scales
    scale0 = get_level_spatial_scale(0)
    scale6 = get_level_spatial_scale(6)
    assert scale0 > scale6  # Coarser levels have larger scales
    print(f"✓ Spatial scale: M0 = {scale0:.0f} km, M6 = {scale6:.0f} km")
    
    # Test edge length distribution
    dist = get_edge_length_distribution(3)
    assert "mean" in dist and "std" in dist
    print(f"✓ Edge distribution at M3: mean = {dist['mean']:.0f} km")
    
    # Test edge classification
    level = classify_edge_by_length(1000)
    assert 2 <= level <= 4  # Should be around M3 (850 km)
    print(f"✓ Edge classification: 1000 km → M{level}")
    
    # Test geodesic distance
    d = geodesic_distance(0, 0, 0, 90)  # 1/4 of equator
    expected = EARTH_CIRCUMFERENCE_KM / 4
    assert abs(d - expected) < 100  # Within 100 km
    print(f"✓ Geodesic distance: (0,0) to (0,90) = {d:.0f} km")
    
    print("\n✅ mesh.py: All tests passed!")
    return True


def test_entropy_module():
    """Test entropy.py functionality."""
    print("\n" + "=" * 60)
    print("Testing: entropy.py")
    print("=" * 60)
    
    from spectral_entropy.entropy import (
        weight_energy,
        spectral_distribution,
        shannon_entropy,
        normalized_entropy,
        spectral_entropy,
        interpret_normalized_entropy,
    )
    
    # Test weight energy
    weights = np.array([1.0, 2.0, 3.0])
    E = weight_energy(weights)
    assert E == 14.0  # 1² + 2² + 3² = 14
    print(f"✓ Weight energy: [1,2,3] → E = {E}")
    
    # Test spectral distribution
    energies = {0: 1.0, 1: 2.0, 2: 1.0}
    p = spectral_distribution(energies)
    assert np.isclose(sum(p), 1.0)
    print(f"✓ Spectral distribution sums to {sum(p):.4f}")
    
    # Test Shannon entropy
    uniform_p = np.array([0.25, 0.25, 0.25, 0.25])
    H = shannon_entropy(uniform_p)
    assert np.isclose(H, np.log(4))  # Maximum entropy
    print(f"✓ Shannon entropy (uniform, n=4): H = {H:.4f} (expected {np.log(4):.4f})")
    
    # Test normalized entropy
    H_n = normalized_entropy(H, 4)
    assert np.isclose(H_n, 1.0)  # Uniform = maximum
    print(f"✓ Normalized entropy: H_n = {H_n:.4f}")
    
    # Test full spectral entropy with synthetic data
    np.random.seed(42)
    level_weights = {
        0: np.random.randn(100) * 1.0,
        1: np.random.randn(200) * 0.8,
        2: np.random.randn(400) * 0.6,
        3: np.random.randn(800) * 0.4,
        4: np.random.randn(1600) * 0.3,
        5: np.random.randn(3200) * 0.2,
        6: np.random.randn(6400) * 0.1,
    }
    
    result = spectral_entropy(level_weights)
    assert 0 <= result.H_normalized <= 1
    print(f"✓ Full spectral entropy: H_n = {result.H_normalized:.4f}")
    print(f"  Dominant scale: M{result.dominant_scale}")
    
    # Test interpretation
    interp = interpret_normalized_entropy(result.H_normalized)
    assert len(interp) > 0
    print(f"✓ Interpretation: {interp[:50]}...")
    
    print("\n✅ entropy.py: All tests passed!")
    return True


def test_power_law_module():
    """Test power_law.py functionality."""
    print("\n" + "=" * 60)
    print("Testing: power_law.py")
    print("=" * 60)
    
    from spectral_entropy.power_law import (
        fit_power_law,
        kolmogorov_reference,
        compare_exponents,
        interpret_exponent,
    )
    from spectral_entropy.mesh import MESH_LEVELS
    
    # Generate synthetic data with known exponent
    np.random.seed(42)
    levels = list(MESH_LEVELS.keys())
    k = np.array([MESH_LEVELS[l].wavenumber for l in levels])
    
    # True power law: E(k) = 0.1 * k^(-0.5)
    true_C = 0.1
    true_alpha = 0.5
    E_true = true_C * k ** (-true_alpha)
    E_noisy = E_true * (1 + 0.05 * np.random.randn(len(k)))
    
    # Test fitting
    fit = fit_power_law(k, E_noisy)
    assert abs(fit.exponent - true_alpha) < 0.2  # Within 0.2 of true
    assert fit.r_squared > 0.9
    print(f"✓ Power law fit: α = {fit.exponent:.4f} (true = {true_alpha})")
    print(f"  R² = {fit.r_squared:.4f}")
    
    # Test Kolmogorov reference
    E_kolmogorov = kolmogorov_reference(k)
    assert len(E_kolmogorov) == len(k)
    print(f"✓ Kolmogorov reference generated")
    
    # Test exponent comparison
    comp = compare_exponents(fit.exponent)
    assert "difference" in comp
    print(f"✓ Comparison: α - 5/3 = {comp['difference']:.4f}")
    
    # Test interpretation
    interp = interpret_exponent(fit.exponent)
    assert len(interp) > 0
    print(f"✓ Interpretation: {interp[:50]}...")
    
    print("\n✅ power_law.py: All tests passed!")
    return True


def test_visualization_module():
    """Test visualize.py functionality (non-display)."""
    print("\n" + "=" * 60)
    print("Testing: visualize.py")
    print("=" * 60)
    
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    
    from spectral_entropy.visualize import (
        set_publication_style,
        plot_energy_spectrum,
        plot_entropy_bars,
    )
    from spectral_entropy.power_law import fit_power_law
    from spectral_entropy.entropy import spectral_entropy
    from spectral_entropy.mesh import MESH_LEVELS
    
    # Generate test data
    np.random.seed(42)
    levels = list(MESH_LEVELS.keys())
    k = np.array([MESH_LEVELS[l].wavenumber for l in levels])
    E = 0.1 * k ** (-0.5) * (1 + 0.05 * np.random.randn(len(k)))
    level_energies = {l: E[i] for i, l in enumerate(levels)}
    
    # Test style setting
    set_publication_style()
    print(f"✓ Publication style set")
    
    # Test energy spectrum plot
    fit = fit_power_law(k, E)
    fig1 = plot_energy_spectrum(k, E, fit)
    assert fig1 is not None
    plt.close(fig1)
    print(f"✓ Energy spectrum plot created")
    
    # Test entropy bars plot
    entropy_result = spectral_entropy(level_energies)
    fig2 = plot_entropy_bars(level_energies, entropy_result)
    assert fig2 is not None
    plt.close(fig2)
    print(f"✓ Entropy bars plot created")
    
    print("\n✅ visualize.py: All tests passed!")
    return True


def test_full_pipeline():
    """Test the complete analysis pipeline."""
    print("\n" + "=" * 60)
    print("Testing: Full Pipeline Integration")
    print("=" * 60)
    
    from spectral_entropy import (
        MESH_LEVELS,
        compute_level_energies,
        spectral_entropy,
        fit_power_law,
        interpret_exponent,
        interpret_normalized_entropy,
    )
    
    # Step 1: Generate synthetic weights (simulating GraphCast)
    print("\n1. Generating synthetic weights...")
    np.random.seed(42)
    level_weights = {}
    
    for level in MESH_LEVELS.keys():
        info = MESH_LEVELS[level]
        # Use same number of weights per level to isolate variance effect
        n_weights = 10000
        # E(k) ~ k^(-0.49), so E ~ L^0.49, and variance ~ L^0.49
        # At level 0 (L=7000), variance is high; at level 6 (L=100), variance is low
        variance = (info.approx_km / 7000) ** 0.49 * 0.01
        level_weights[level] = np.random.randn(n_weights) * np.sqrt(variance)
    
    print(f"   Generated weights for {len(level_weights)} levels")
    
    # Step 2: Compute energies
    print("\n2. Computing weight energies...")
    level_energies = compute_level_energies(level_weights)
    total_energy = sum(level_energies.values())
    print(f"   Total energy: {total_energy:.6f}")
    
    # Step 3: Compute spectral entropy
    print("\n3. Computing spectral entropy...")
    result = spectral_entropy(level_weights)
    print(f"   H_s = {result.H_raw:.4f} nats")
    print(f"   H_n = {result.H_normalized:.4f}")
    
    # Step 4: Fit power law
    print("\n4. Fitting power law...")
    levels = sorted(level_energies.keys())
    k = np.array([MESH_LEVELS[l].wavenumber for l in levels])
    E = np.array([level_energies[l] for l in levels])
    fit = fit_power_law(k, E)
    print(f"   E(k) = {fit.amplitude:.4e} × k^(-{fit.exponent:.4f})")
    print(f"   R² = {fit.r_squared:.4f}")
    
    # Step 5: Interpret results
    print("\n5. Interpreting results...")
    print(f"   Exponent: {interpret_exponent(fit.exponent)[:60]}...")
    print(f"   Entropy: {interpret_normalized_entropy(result.H_normalized)[:60]}...")
    
    # Validate results are in expected ranges
    assert 0.3 < fit.exponent < 0.7, f"Exponent {fit.exponent} outside expected range"
    assert 0.7 < result.H_normalized < 1.0, f"Entropy {result.H_normalized} outside expected range"
    assert fit.r_squared > 0.8, f"R² {fit.r_squared} too low"
    
    print("\n✅ Full Pipeline: All tests passed!")
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("SPECTRAL ENTROPY ANALYSIS - TEST SUITE")
    print("=" * 60)
    
    results = {}
    
    try:
        results["mesh"] = test_mesh_module()
    except Exception as e:
        print(f"\n❌ mesh.py failed: {e}")
        results["mesh"] = False
    
    try:
        results["entropy"] = test_entropy_module()
    except Exception as e:
        print(f"\n❌ entropy.py failed: {e}")
        results["entropy"] = False
    
    try:
        results["power_law"] = test_power_law_module()
    except Exception as e:
        print(f"\n❌ power_law.py failed: {e}")
        results["power_law"] = False
    
    try:
        results["visualize"] = test_visualization_module()
    except Exception as e:
        print(f"\n❌ visualize.py failed: {e}")
        results["visualize"] = False
    
    try:
        results["pipeline"] = test_full_pipeline()
    except Exception as e:
        print(f"\n❌ Full pipeline failed: {e}")
        results["pipeline"] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {name:<15} {status}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    if all_passed:
        print("All tests passed! ✅")
        return 0
    else:
        print("Some tests failed. ❌")
        return 1


if __name__ == "__main__":
    sys.exit(main())
