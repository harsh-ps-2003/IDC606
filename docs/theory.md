# Theoretical Framework: From Shannon Entropy to Spectral Analysis

This document provides the theoretical foundation for analyzing spectral entropy in hierarchical neural networks like GraphCast, bridging classical information theory with fluid dynamics turbulence analogies.

## Table of Contents

1. [Classical Shannon Entropy](#1-classical-shannon-entropy)
2. [Spectral Extension](#2-spectral-extension)
3. [Turbulence Analogy](#3-turbulence-analogy)
4. [GraphCast Mapping](#4-graphcast-mapping)
5. [Mathematical Formulations](#5-mathematical-formulations)
6. [Physical Interpretation](#6-physical-interpretation)

---

## 1. Classical Shannon Entropy

### Definition

Shannon entropy, introduced by Claude Shannon in 1948, quantifies the average information content (or uncertainty) in a random variable. For a discrete probability distribution P = {p₁, p₂, ..., pₙ}, the entropy is:

$$H = -\sum_{i=1}^{n} p_i \log(p_i)$$

### Properties

- **Non-negativity**: H ≥ 0
- **Maximum**: H_max = log(n) when p_i = 1/n (uniform distribution)
- **Minimum**: H = 0 when p_j = 1 for some j (deterministic)
- **Additivity**: H(X,Y) = H(X) + H(Y) for independent X, Y

### Information-Theoretic Interpretation

- **High entropy**: High uncertainty, uniform distribution, maximum information content
- **Low entropy**: Low uncertainty, concentrated distribution, predictable outcomes

### Units

- **Nats**: Using natural logarithm (ln)
- **Bits**: Using log base 2 (log₂)
- **Dits**: Using log base 10 (log₁₀)

---

## 2. Spectral Extension

### Redefining Probability as Spectral Weight

In the spectral framework, we reinterpret the probability distribution in terms of **energy density** across scales:

$$p_r = \frac{E_r}{\sum_j E_j}$$

where:
- E_r is the "energy" at scale r
- The sum normalizes to create a valid probability distribution

### Energy Definition for Neural Networks

For neural network weights at level r:

$$E_r = \sum_{w \in W_r} w^2$$

This is analogous to:
- **Turbulence**: Kinetic energy = ½ρv²
- **Signal processing**: Power spectral density
- **Statistical mechanics**: Energy of a microstate

### Spectral Entropy

Applying Shannon's formula to the spectral distribution:

$$H_s = -\sum_{r=0}^{R} p_r \ln(p_r)$$

### Normalized Spectral Entropy

To enable comparison across systems with different numbers of scales:

$$H_n = \frac{H_s}{\ln(R+1)} \in [0, 1]$$

where R+1 is the number of levels.

### Interpretation

| H_n Value | Interpretation |
|-----------|----------------|
| ≈ 0 | Energy concentrated at one scale |
| ≈ 0.5 | Partial scale localization |
| ≈ 0.88 | High multiscale complexity (GraphCast) |
| ≈ 1.0 | Uniform energy across all scales |

---

## 3. Turbulence Analogy

### Kolmogorov's Energy Cascade

In 1941, Kolmogorov proposed a universal theory for fully developed turbulence. Energy is injected at large scales (low k) and cascades to small scales (high k) where it dissipates.

The energy spectrum follows:

$$E(k) \sim \varepsilon^{2/3} k^{-5/3}$$

where:
- k is the wavenumber (1/length)
- ε is the energy dissipation rate
- The exponent -5/3 ≈ -1.667 is universal

### Scaling Regimes

| Region | Scaling | Physical Process |
|--------|---------|------------------|
| Energy injection | Variable | External forcing |
| Inertial range | k^(-5/3) | Conservative energy transfer |
| Dissipation range | Exponential decay | Viscous dissipation |

### 2D Turbulence

In two dimensions, there are two cascades:

1. **Inverse energy cascade**: E(k) ~ k^(-5/3) (toward large scales)
2. **Direct enstrophy cascade**: E(k) ~ k^(-3) (toward small scales)

### Application to Neural Networks

We hypothesize that information in hierarchical neural networks follows an analogous "cascade" from large to small scales, but with different exponents reflecting the nature of learned representations.

---

## 4. GraphCast Mapping

### Multi-Mesh Architecture

GraphCast uses an icosahedral mesh refined 6 times:

| Level | Mesh | Nodes | Edges | Approx. Scale (km) |
|-------|------|-------|-------|-------------------|
| 0 | M₀ | 12 | 60 | ~7,000 |
| 1 | M₁ | 42 | 240 | ~3,500 |
| 2 | M₂ | 162 | 960 | ~1,700 |
| 3 | M₃ | 642 | 3,840 | ~850 |
| 4 | M₄ | 2,562 | 15,360 | ~425 |
| 5 | M₅ | 10,242 | 61,440 | ~212 |
| 6 | M₆ | 40,962 | 245,760 | ~100 |

### Wavenumber Mapping

The wavenumber at each level is:

$$k_r = \frac{1}{L_r}$$

where L_r is the characteristic length scale at level r.

For icosahedral meshes:

$$L_r \approx \frac{C_{\text{Earth}}}{5 \times 2^r}$$

where C_Earth ≈ 40,075 km.

### Physical Analogs

| Level | Scale | Atmospheric Analog |
|-------|-------|-------------------|
| M₀ | ~7,000 km | Planetary waves, Hadley cell |
| M₁ | ~3,500 km | Synoptic-scale flow |
| M₂ | ~1,700 km | Large-scale pressure systems |
| M₃ | ~850 km | Frontal systems |
| M₄ | ~425 km | Regional weather patterns |
| M₅ | ~212 km | Mesoscale convection |
| M₆ | ~100 km | Local turbulence |

---

## 5. Mathematical Formulations

### Complete Spectral Entropy Framework

**Input**: Weight matrices {W_r} for r = 0, 1, ..., R

**Step 1**: Compute energy at each level
$$E_r = \sum_{w \in W_r} w^2$$

**Step 2**: Normalize to probability distribution
$$p_r = \frac{E_r}{\sum_{j=0}^{R} E_j}$$

**Step 3**: Compute Shannon entropy
$$H_s = -\sum_{r=0}^{R} p_r \ln(p_r)$$

**Step 4**: Normalize
$$H_n = \frac{H_s}{\ln(R+1)}$$

### Power Law Fitting

Fit the energy spectrum to:
$$E(k) = C \cdot k^{-\alpha}$$

In log-log space:
$$\ln(E) = \ln(C) - \alpha \ln(k)$$

This is a linear regression with:
- Slope = -α
- Intercept = ln(C)

### Goodness of Fit

The coefficient of determination:
$$R^2 = 1 - \frac{SS_{res}}{SS_{tot}}$$

where:
- SS_res = Σ(E_i - Ê_i)² (residual sum of squares)
- SS_tot = Σ(E_i - Ē)² (total sum of squares)

---

## 6. Physical Interpretation

### The k^(-0.49) Finding

GraphCast exhibits E(k) ~ k^(-0.49), which is significantly shallower than Kolmogorov's -5/3 ≈ -1.67.

**Implications**:

1. **More energy at small scales**: The shallow slope means information is preserved better at fine scales compared to physical turbulence.

2. **Lower "informational viscosity"**: In turbulence, viscosity causes rapid energy dissipation at small scales. The shallow exponent suggests the neural network acts as a low-viscosity medium for information.

3. **Broadband participation**: High normalized entropy (H_n ≈ 0.88) indicates all scales contribute meaningfully to the representation.

### Comparison Table

| Exponent α | Physical System |
|------------|-----------------|
| 0.49 | GraphCast neural network |
| 1.0 | Batchelor (passive scalar) |
| 5/3 ≈ 1.67 | Kolmogorov 3D turbulence |
| 2.0 | Burgers (shock-dominated) |
| 3.0 | 2D enstrophy cascade |

### The k^(-1/2) Bridge

The exponent α ≈ 0.5 suggests a state between:

- **White noise** (α = 0): Completely uncorrelated, uniform power
- **Kolmogorov turbulence** (α = 5/3): Strong scale correlations

This "weakly correlated" state maintains structure while allowing information to persist across scales—arguably ideal for a weather prediction network that must capture both global circulation and local weather events.

---

## References

1. Shannon, C.E. (1948). "A Mathematical Theory of Communication." *Bell System Technical Journal*, 27(3), 379-423.

2. Kolmogorov, A.N. (1941). "The local structure of turbulence in incompressible viscous fluid for very large Reynolds numbers." *Dokl. Akad. Nauk SSSR*, 30, 301-305.

3. Kraichnan, R.H. (1967). "Inertial ranges in two-dimensional turbulence." *Physics of Fluids*, 10(7), 1417-1423.

4. Lam, R., et al. (2022). "GraphCast: Learning skillful medium-range global weather forecasting." *arXiv:2212.12794*.

5. Verma, M.K. (2019). *Energy Transfers in Fluid Flows: Multiscale and Spectral Perspectives*. Cambridge University Press.
