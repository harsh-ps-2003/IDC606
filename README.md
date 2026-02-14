# Spectral Entropy Analysis for GraphCast

A framework for analyzing the spectral entropy of multi-mesh neural network weights in GraphCast, 
drawing analogies between information theory and fluid dynamics turbulence cascades.

## Overview

This project treats GraphCast's multi-mesh hierarchy as an "information cascade" analogous to 
Kolmogorov's energy cascade in turbulence. By computing the spectral entropy of neural network 
weights across different mesh refinement levels, we can characterize the "information health" 
and structural complexity of hierarchical weather forecasting models.

### Key Findings

- **Power Law Exponent**: E(k) ~ k^(-0.49), significantly different from Kolmogorov's -5/3
- **Normalized Spectral Entropy**: H_n ≈ 0.88, indicating high multiscale complexity
- **Physical Interpretation**: The shallow exponent suggests lower "informational viscosity"

## Installation

```bash
# Clone and install
cd IDC606
pip install -e .

# Or install dependencies directly
pip install -r requirements.txt
```

## Quick Start

```python
from spectral_entropy import (
    load_graphcast_params,
    extract_processor_weights,
    spectral_entropy,
    fit_power_law,
    plot_energy_spectrum
)

# Load GraphCast weights
params = load_graphcast_params("0.25deg")
level_weights = extract_processor_weights(params)

# Compute spectral entropy
entropy_result = spectral_entropy(level_weights)
print(f"Normalized Entropy: {entropy_result['H_normalized']:.4f}")

# Fit power law
k, E = entropy_result['wavenumbers'], entropy_result['energies']
fit = fit_power_law(k, E)
print(f"Power law: E(k) ~ k^({fit.exponent:.2f}), R² = {fit.r_squared:.4f}")

# Visualize
plot_energy_spectrum(k, E, fit, save_path="energy_spectrum.png")
```

## Project Structure

```
IDC606/
├── spectral_entropy/           # Python module
│   ├── __init__.py
│   ├── mesh.py                 # Multi-mesh geometry utilities
│   ├── extractor.py            # Weight extraction from GraphCast
│   ├── entropy.py              # Shannon & spectral entropy
│   ├── power_law.py            # Power law fitting (k^-α)
│   └── visualize.py            # Log-log plots, comparisons
├── notebooks/
│   └── graphcast_spectral_analysis.ipynb
├── docs/
│   ├── theory.md               # Full theoretical framework
│   └── physical_vs_informational.md
├── data/                       # Downloaded weights cache
├── requirements.txt
└── pyproject.toml
```

## Multi-Mesh Hierarchy

GraphCast uses a hierarchical icosahedral mesh with 7 refinement levels:

| Level | Nodes   | Edges    | Approx. Scale | Physical Analog            |
|-------|---------|----------|---------------|----------------------------|
| M₀    | 12      | 60       | ~7,000 km     | Planetary waves            |
| M₁    | 42      | 240      | ~3,500 km     | Synoptic background flow   |
| M₂    | 162     | 960      | ~1,700 km     | Large-scale vortices       |
| M₃    | 642     | 3,840    | ~850 km       | Frontal systems            |
| M₄    | 2,562   | 15,360   | ~425 km       | Regional moisture transport|
| M₅    | 10,242  | 61,440   | ~212 km       | Mesoscale convective       |
| M₆    | 40,962  | 245,760  | ~100 km       | Localized turbulence       |

## Mathematical Framework

### Spectral Energy
$$E_r = \sum_{w \in W_r} w^2$$

### Wavenumber
$$k_r = \frac{1}{L_r} \quad \text{where } L_r \approx \frac{C_{\text{Earth}}}{5 \times 2^r}$$

### Spectral Entropy
$$H_s = -\sum_{r=0}^{R} p_r \ln(p_r), \quad p_r = \frac{E_r}{\sum_j E_j}$$

$$H_n = \frac{H_s}{\ln(R+1)} \in [0, 1]$$

## References

1. Lam, R., et al. (2022). "GraphCast: Learning skillful medium-range global weather forecasting." 
   arXiv:2212.12794

2. Kolmogorov, A.N. (1941). "The local structure of turbulence in incompressible viscous fluid 
   for very large Reynolds numbers."

3. Shannon, C.E. (1948). "A Mathematical Theory of Communication."

## License

MIT License
