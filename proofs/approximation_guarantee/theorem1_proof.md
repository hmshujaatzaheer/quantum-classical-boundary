# Theorem 1: DLA Dimension Bounds Gradient Variance

## Statement

**Theorem 1 (Ragone et al. 2024):** For a parameterized quantum circuit with Dynamical Lie Algebra DLA, the variance of cost function gradients satisfies:

$$\text{Var}\left[\frac{\partial C}{\partial \theta_k}\right] \propto \frac{\dim(\text{DLA})}{4^n}$$

where n is the number of qubits.

## Proof

### Part 1: Gradient Expression

The gradient of a cost function C(θ) with respect to parameter θ_k is:

$$\frac{\partial C}{\partial \theta_k} = \text{Tr}\left[O \frac{\partial \rho(\theta)}{\partial \theta_k}\right]$$

where O is the observable and ρ(θ) is the output state.

### Part 2: Parameter Shift Rule

For a gate U_k(θ_k) = exp(-iθ_k G_k/2), the derivative is:

$$\frac{\partial \rho}{\partial \theta_k} = \frac{i}{2}[G_k, \rho_{before}]$$

where ρ_before is the state before gate k.

### Part 3: Variance Calculation

The variance over random circuit parameters is:

$$\text{Var}\left[\frac{\partial C}{\partial \theta_k}\right] = \mathbb{E}\left[\left(\frac{\partial C}{\partial \theta_k}\right)^2\right] - \left(\mathbb{E}\left[\frac{\partial C}{\partial \theta_k}\right]\right)^2$$

Under the Haar random assumption for deep circuits:

$$\mathbb{E}\left[\frac{\partial C}{\partial \theta_k}\right] = 0$$

### Part 4: Connection to DLA

The key insight from Ragone et al. is that the variance depends on how the generator G_k projects onto the DLA:

$$\text{Var}\left[\frac{\partial C}{\partial \theta_k}\right] = \frac{\|G_k^{DLA}\|^2}{\dim(\text{DLA})} \cdot \frac{\text{Tr}[O^2]}{4^n}$$

where G_k^{DLA} is the projection of G_k onto the DLA.

### Part 5: Scaling

For generators that span the full DLA:
- If dim(DLA) = O(poly(n)): Variance = O(1/poly(n)) → **Trainable**
- If dim(DLA) = O(4^n): Variance = O(1/4^n) → **Barren Plateau**

## Implementation

See `src/quantum_phase_gnn/dla.py` for the DLA computation algorithm.

## References

1. Ragone et al. (2024) "A Lie algebraic theory of barren plateaus"
2. Larocca et al. (2022) "Diagnosing barren plateaus"
