# Documentation

This directory contains the theoretical background and detailed analysis documentation for the Spectral Entropy Analysis framework.

## Contents

### [theory.md](theory.md)

Comprehensive theoretical framework covering:
- Classical Shannon entropy foundations
- Spectral extension to energy distributions
- Kolmogorov turbulence analogy
- GraphCast multi-mesh mapping
- Complete mathematical formulations
- Physical interpretation guide

### [physical_vs_informational.md](physical_vs_informational.md)

Detailed comparison between:
- Shannon entropy (classical information theory)
- Spectral entropy (turbulence-inspired framework)
- Feature-by-feature comparison table
- k^(-1/2) power law interpretation
- Full mesh hierarchy with physical analogs

## Quick Navigation

| Topic | Document | Section |
|-------|----------|---------|
| What is Shannon entropy? | [theory.md](theory.md) | §1 |
| How is spectral entropy different? | [theory.md](theory.md) | §2 |
| What is Kolmogorov turbulence? | [theory.md](theory.md) | §3 |
| How does GraphCast's mesh work? | [theory.md](theory.md) | §4 |
| Mathematical formulas | [theory.md](theory.md) | §5 |
| Shannon vs Spectral comparison | [physical_vs_informational.md](physical_vs_informational.md) | §3 |
| k^(-1/2) interpretation | [physical_vs_informational.md](physical_vs_informational.md) | §4 |
| Mesh level physical analogs | [physical_vs_informational.md](physical_vs_informational.md) | §5 |

## Key Equations

### Shannon Entropy
```
H = -Σ pᵢ log(pᵢ)
```

### Spectral Distribution
```
pᵣ = Eᵣ / ΣEⱼ,  where Eᵣ = Σw²
```

### Power Law
```
E(k) = C × k^(-α)
```

### Normalized Entropy
```
Hₙ = Hₛ / ln(n) ∈ [0, 1]
```

## Key Findings

| Metric | Value | Significance |
|--------|-------|--------------|
| Power law exponent (α) | 0.49 | Shallower than Kolmogorov's 5/3 |
| Normalized entropy (Hₙ) | 0.88 | High multiscale complexity |
| R² of fit | 0.93 | Robust power law behavior |
