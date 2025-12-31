# Theorem 3: Sample Complexity of BELL-VERIFY-NEAR-TERM

## Statement

**Theorem 3:** The BELL-VERIFY-NEAR-TERM protocol requires:

$$N = O\left(\frac{\log(1/δ)}{ε^2}\right)$$

samples to achieve fidelity estimate within ±ε with probability ≥ 1-δ.

## Proof

### Part 1: XEB Score Estimator

The linear XEB score is estimated as:

$$\hat{F}_{XEB} = 2^n · \frac{1}{N}\sum_{i=1}^N p(x_i) - 1$$

where x_i are measured bitstrings and p(x) are ideal probabilities.

### Part 2: Variance Analysis

The variance of the estimator is:

$$\text{Var}[\hat{F}_{XEB}] = \frac{4^n}{N} \cdot \text{Var}[p(x)] ≤ \frac{1}{N}$$

### Part 3: Hoeffding Bound

By Hoeffding's inequality:

$$\Pr[|\hat{F} - F| > ε] ≤ 2\exp(-2Nε^2)$$

### Part 4: Sample Complexity

Setting the RHS ≤ δ:

$$N ≥ \frac{\log(2/δ)}{2ε^2} = O\left(\frac{\log(1/δ)}{ε^2}\right)$$

## Implementation

See `src/bell_verify/protocol.py` for the verification algorithm.

## Practical Considerations

For typical parameters (ε = 0.01, δ = 0.05):
- N ≈ 15,000 samples required
- Runtime: ~15 seconds on typical NISQ device
