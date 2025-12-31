# Convergence Analysis for POSITIVITY-GAUGE-OPT

## Statement

**Theorem (Convergence):** Let P: GL(D)^|E| → ℝ be the positivity loss function. With step size η_t = O(1/√t), the Riemannian gradient descent in POSITIVITY-GAUGE-OPT satisfies:

$$\min_{t≤T} \|∇P(G_t)\|^2 ≤ O\left(\frac{P(G_0) - P^*}{\sqrt{T}}\right)$$

## Proof

### Part 1: Riemannian Gradient Descent

On the GL(D) manifold, the update rule is:

$$G_{t+1} = R_{G_t}(-η_t · ∇_G P)$$

where R is the retraction operator.

### Part 2: Descent Lemma

For L-smooth loss on the manifold:

$$P(G_{t+1}) ≤ P(G_t) - η_t \|∇P(G_t)\|^2 + \frac{L η_t^2}{2}\|∇P(G_t)\|^2$$

### Part 3: Telescoping Sum

Summing over iterations with η_t = 1/(L√t):

$$\sum_{t=1}^T \frac{1}{2√t}\|∇P(G_t)\|^2 ≤ P(G_0) - P^*$$

### Part 4: Convergence Rate

This gives:

$$\min_{t≤T} \|∇P(G_t)\|^2 ≤ O\left(\frac{P(G_0) - P^*}{\sqrt{T}}\right)$$

## Implementation

See `src/positivity_gauge_opt/gauge_optimizer.py` for the optimization loop.
