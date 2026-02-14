# Physical vs. Informational Entropy

This document explores the relationship between classical Shannon entropy and the spectral entropy framework applied to neural network weight analysis, with specific application to GraphCast's multi-mesh architecture.

## 1. Spectral Weight vs. Probability

### Classical Shannon Formula

In Shannon's original formulation:

$$H = -\sum_i p_i \ln(p_i)$$

The p_i values are **probabilities** of discrete outcomes, summing to 1.

### Spectral Reinterpretation

In our framework, we replace abstract probabilities with **physical quantities**:

$$p_i \rightarrow \frac{E(k_i)}{\sum_j E(k_j)}$$

where:
- E(k_i) = Σw² is the "weight energy" at wavenumber k_i
- The normalization ensures Σp_i = 1

This substitution preserves the mathematical structure of entropy while adding **physical meaning** to each term.

---

## 2. Topological Constraints

### Scale-Bound Entropy

Unlike abstract Shannon entropy, spectral entropy is **anchored to physical scales**:

| Level | Distance (km) | Wavenumber k (1/km) | Physical Meaning |
|-------|---------------|---------------------|------------------|
| M₀ | ~7,000 | 1.4 × 10⁻⁴ | Planetary circulation |
| M₁ | ~3,500 | 2.9 × 10⁻⁴ | Synoptic patterns |
| M₂ | ~1,700 | 5.9 × 10⁻⁴ | Large pressure systems |
| M₃ | ~850 | 1.2 × 10⁻³ | Frontal systems |
| M₄ | ~425 | 2.4 × 10⁻³ | Regional weather |
| M₅ | ~212 | 4.7 × 10⁻³ | Mesoscale convection |
| M₆ | ~100 | 1.0 × 10⁻² | Local turbulence |

### Dynamical Complexity

Spectral entropy measures **dynamical complexity**—how uniformly the neural network distributes its representational capacity across spatial scales.

- **High H_n**: The network treats all scales as important
- **Low H_n**: The network focuses on specific scales (potentially ignoring others)

For weather prediction, high spectral entropy is desirable because atmospheric dynamics span all these scales simultaneously.

---

## 3. Comparison Framework

### Feature Comparison Table

| Feature | Shannon Entropy (Classical) | Spectral Entropy (This Work) |
|---------|----------------------------|------------------------------|
| **Primary Goal** | Minimize uncertainty / Maximize compression | Characterize multiscale information structure |
| **p_i Interpretation** | Probability of symbol occurrence | Normalized energy at scale L_i |
| **System State** | Static probability distribution | Dynamic "energy cascade" through scales |
| **Ideal Value** | Low (for efficiency/compression) | High (for physical realism in prediction) |
| **Scale Awareness** | None—abstract symbols | Explicit—each p_i has physical dimension |
| **Analogous Field** | Coding theory, compression | Turbulence theory, spectral analysis |
| **Typical Units** | Bits per symbol | Nats (or bits) over spatial spectrum |
| **Normalization** | H ∈ [0, log(n)] | H_n ∈ [0, 1] |

### When to Use Each

**Shannon Entropy**: 
- Text compression
- Source coding
- Channel capacity
- Abstract discrete distributions

**Spectral Entropy**:
- Analyzing hierarchical neural networks
- Characterizing multiscale physical simulations
- Comparing model architectures
- Understanding learned representations

---

## 4. The k^(-1/2) Power Law Interpretation

### Observed Scaling

GraphCast's weight energy spectrum follows:

$$E(k) \approx 0.023 \cdot k^{-0.49}$$

with R² = 0.93, indicating a robust fit.

### Comparison to Known Regimes

| Exponent α | Physical Regime | Correlation Structure |
|------------|-----------------|----------------------|
| 0 | White noise | Uncorrelated |
| **0.49** | **GraphCast** | **Weakly correlated** |
| 1.0 | Batchelor | Passive scalar mixing |
| 5/3 ≈ 1.67 | Kolmogorov | Strongly correlated (3D turbulence) |
| 3.0 | Enstrophy | Very strongly correlated (2D turbulence) |

### The "Bridge" Interpretation

The k^(-1/2) scaling represents a **bridge** between:

1. **Pure randomness** (α = 0): No scale structure, all wavenumbers equally energetic
2. **Turbulent cascade** (α = 5/3): Strong scale coupling, energy systematically transferred

GraphCast's intermediate exponent suggests:
- Information flows between scales but without strict energy conservation
- The network maintains detail at small scales better than physical turbulence
- A "low viscosity" information medium that preserves fine structure

---

## 5. Full Hierarchy Table

### Mesh Level to Physical/Hydrodynamic Analog

| Level | Approx. Distance | Weight Energy (Σw²) | Physical/Hydrodynamic Analog |
|-------|------------------|---------------------|------------------------------|
| M₀ | ~7,000+ km | Variable | **Planetary forcing**: Hadley circulation, Rossby waves, global teleconnections |
| M₁ | ~3,500 km | Variable | **Synoptic background**: Jet streams, large-scale advection |
| M₂ | ~1,700 km | Variable | **Baroclinic scale**: Major cyclones, blocking patterns |
| M₃ | ~850 km | Variable | **Inertial range start**: Frontal dynamics, extratropical storms |
| M₄ | ~425 km | Variable | **Deep inertial range**: Mesoscale lows, squall lines |
| M₅ | ~212 km | Variable | **Small mesoscale**: MCSs, lake-effect snow, orographic flows |
| M₆ | ~100 km | Variable | **Dissipative/subgrid**: Boundary layer eddies, cumulus |

### Interpretation Notes

1. **M₀-M₁ (Planetary)**: These levels capture global-scale forcing and large-scale flow patterns that set the stage for regional weather.

2. **M₂-M₃ (Synoptic)**: The classical weather map scales—pressure systems, fronts, and storm tracks.

3. **M₄-M₅ (Mesoscale)**: Features often missed by coarse NWP models—convective systems, terrain effects, local circulations.

4. **M₆ (Boundary Layer)**: The smallest resolved scale, capturing processes that would be parameterized in traditional models.

---

## 6. Implications for Neural Weather Prediction

### Why High Spectral Entropy Matters

1. **Complete Representation**: High H_n indicates the network represents all scales, not just dominant ones.

2. **Physical Fidelity**: Weather systems span the full scale range; a model biased toward certain scales will fail at others.

3. **Generalization**: Uniform scale coverage suggests better generalization to unusual weather patterns.

### Why k^(-0.5) Differs from k^(-5/3)

The Kolmogorov cascade assumes:
- Conservative energy transfer
- Local interactions in wavenumber space
- Statistical stationarity

Neural networks violate these assumptions:
- Information, not energy, is being processed
- Connections can skip scales (non-local in k-space)
- Training is inherently non-stationary

The shallow exponent reflects these fundamental differences.

---

## 7. Mathematical Summary

### Spectral Entropy Calculation

$$H_s = -\sum_{r=0}^{6} p_r \ln(p_r)$$

where

$$p_r = \frac{E_r}{\sum_{j=0}^{6} E_j}, \quad E_r = \sum_{w \in W_r} w^2$$

### Normalized Form

$$H_n = \frac{H_s}{\ln(7)} \approx \frac{H_s}{1.946}$$

### Your Findings

| Metric | Value | Interpretation |
|--------|-------|----------------|
| H_s | 2.0259 nats | High absolute entropy |
| H_n | 0.8798 | 88% of maximum possible entropy |
| α | 0.49 | Shallow spectral slope |
| R² | 0.93 | Good power law fit |

---

## 8. Conclusion

The spectral entropy framework provides a physically meaningful way to analyze hierarchical neural network architectures. By mapping abstract weight distributions to spatial scales, we gain insight into:

1. **Scale coverage**: Which scales the network prioritizes
2. **Information cascade**: How information flows between scales
3. **Physical analogy**: Connections to turbulence theory

GraphCast's high normalized entropy (H_n ≈ 0.88) and shallow spectral slope (α ≈ 0.49) indicate a well-balanced architecture that maintains representational capacity across all atmospheric scales—a desirable property for a general-purpose weather prediction model.

---

## References

1. Shannon, C.E. (1948). "A Mathematical Theory of Communication."

2. Kolmogorov, A.N. (1941). "The local structure of turbulence."

3. Lam, R., et al. (2022). "GraphCast: Learning skillful medium-range global weather forecasting."

4. Verma, M.K. (2019). "Energy Transfers in Fluid Flows."

5. Frisch, U. (1995). "Turbulence: The Legacy of A.N. Kolmogorov."
