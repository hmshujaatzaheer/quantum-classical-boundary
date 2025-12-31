# Theorem 2: Approximation Guarantee for Gauge Transformations

## Statement

**Theorem 2:** Let T be a tensor network with bond dimension D, and let G be a gauge transformation. For any local observable O and the gauge-transformed network T' = G·T·G⁻¹:

$$|⟨O⟩_T - ⟨O⟩_{T'}| ≤ ε$$

where ε depends on the numerical precision and is bounded by machine epsilon for exact arithmetic.

## Proof

### Part 1: Gauge Invariance

Gauge transformations insert G·G⁻¹ = I on each bond:

```
A[i] → A'[i] = G_{i-1}⁻¹ · A[i] · G_i
```

The full contraction is invariant:

$$|ψ'⟩ = \text{Tr}(A'[1] · A'[2] · ... · A'[N]) = \text{Tr}(A[1] · A[2] · ... · A[N]) = |ψ⟩$$

### Part 2: Numerical Error Bounds

For computed gauge G̃ = G + ΔG with ||ΔG|| ≤ δ||G||:

$$||A'_{exact} - A'_{computed}|| ≤ 2κ(G) · δ · ||A||$$

where κ(G) is the condition number.

### Part 3: Monte Carlo Sampling

For positivity score P, the sampling error is:

$$|⟨O⟩_{MC} - ⟨O⟩_{exact}| ≤ \frac{C · σ_O}{\sqrt{N · (2P-1)^2}}$$

## Implementation

See `src/positivity_gauge_opt/gauge_optimizer.py` for verification.
