"""
Riemannian Optimization on GL(D) Manifold
=========================================

Implements Riemannian gradient descent for gauge optimization on the
General Linear group GL(D) of invertible matrices.

This module provides the manifold optimization infrastructure for
POSITIVITY-GAUGE-OPT (Algorithm 3).

Author: PhD Research - ETH Zurich Quantum Computing Group
"""

import numpy as np
from typing import Tuple, Callable, Optional, List
from dataclasses import dataclass


@dataclass
class RiemannianOptimizationResult:
    """Result of Riemannian optimization."""
    optimal_point: np.ndarray
    optimal_value: float
    gradient_norm_history: List[float]
    value_history: List[float]
    iterations: int
    converged: bool


class GLManifold:
    """
    The General Linear group GL(D) of D×D invertible matrices.
    
    This is an open subset of R^(D×D), making it a smooth manifold.
    The tangent space at any point G is the full matrix space R^(D×D).
    """
    
    def __init__(self, dimension: int):
        """
        Initialize GL(D) manifold.
        
        Parameters
        ----------
        dimension : int
            Matrix dimension D
        """
        self.D = dimension
    
    def random_point(self, seed: Optional[int] = None) -> np.ndarray:
        """Generate random point on GL(D) (random invertible matrix)."""
        if seed is not None:
            np.random.seed(seed)
        # Start with identity + small perturbation to ensure invertibility
        G = np.eye(self.D) + 0.1 * np.random.randn(self.D, self.D)
        return G
    
    def project_to_manifold(self, G: np.ndarray) -> np.ndarray:
        """
        Project matrix to GL(D) by ensuring invertibility.
        
        Uses SVD to handle near-singular matrices.
        """
        U, S, Vh = np.linalg.svd(G)
        # Clamp singular values away from zero
        S = np.maximum(S, 0.01)
        return U @ np.diag(S) @ Vh
    
    def is_on_manifold(self, G: np.ndarray, tol: float = 1e-10) -> bool:
        """Check if matrix is invertible (on GL(D))."""
        return np.abs(np.linalg.det(G)) > tol
    
    def tangent_space_dimension(self) -> int:
        """Dimension of tangent space (D²)."""
        return self.D * self.D


class OrthogonalManifold:
    """
    The Orthogonal group O(D) of D×D orthogonal matrices.
    
    Submanifold of GL(D) with constraint G^T G = I.
    """
    
    def __init__(self, dimension: int):
        self.D = dimension
    
    def random_point(self, seed: Optional[int] = None) -> np.ndarray:
        """Generate random orthogonal matrix using QR decomposition."""
        if seed is not None:
            np.random.seed(seed)
        A = np.random.randn(self.D, self.D)
        Q, _ = np.linalg.qr(A)
        return Q
    
    def project_to_manifold(self, G: np.ndarray) -> np.ndarray:
        """Project to O(D) using polar decomposition."""
        U, _, Vh = np.linalg.svd(G)
        return U @ Vh
    
    def project_to_tangent(self, G: np.ndarray, V: np.ndarray) -> np.ndarray:
        """
        Project vector V to tangent space at G.
        
        Tangent space of O(D) at G consists of matrices V such that
        G^T V is skew-symmetric.
        """
        # V_tangent = V - G @ sym(G^T V) where sym(A) = (A + A^T)/2
        GtV = G.T @ V
        symmetric_part = (GtV + GtV.T) / 2
        return V - G @ symmetric_part
    
    def is_on_manifold(self, G: np.ndarray, tol: float = 1e-6) -> bool:
        """Check if matrix is orthogonal."""
        return np.allclose(G.T @ G, np.eye(self.D), atol=tol)


class RiemannianGradientDescent:
    """
    Riemannian gradient descent optimizer.
    
    Implements gradient descent on Riemannian manifolds using
    retractions instead of standard addition.
    """
    
    def __init__(self, manifold: GLManifold, learning_rate: float = 0.1,
                 max_iterations: int = 1000, tolerance: float = 1e-6,
                 momentum: float = 0.0):
        """
        Initialize optimizer.
        
        Parameters
        ----------
        manifold : GLManifold or OrthogonalManifold
            The manifold to optimize on
        learning_rate : float
            Step size for gradient descent
        max_iterations : int
            Maximum number of iterations
        tolerance : float
            Convergence tolerance for gradient norm
        momentum : float
            Momentum coefficient (0 = no momentum)
        """
        self.manifold = manifold
        self.lr = learning_rate
        self.max_iter = max_iterations
        self.tol = tolerance
        self.momentum = momentum
    
    def optimize(self, objective: Callable[[np.ndarray], float],
                 gradient: Callable[[np.ndarray], np.ndarray],
                 initial_point: Optional[np.ndarray] = None,
                 verbose: bool = False) -> RiemannianOptimizationResult:
        """
        Minimize objective function on manifold.
        
        Parameters
        ----------
        objective : Callable
            Function to minimize: f(G) -> R
        gradient : Callable
            Euclidean gradient of objective: ∇f(G) -> R^(D×D)
        initial_point : np.ndarray, optional
            Starting point (random if not provided)
        verbose : bool
            Print progress
            
        Returns
        -------
        RiemannianOptimizationResult
            Optimization result with optimal point and history
        """
        # Initialize
        if initial_point is None:
            G = self.manifold.random_point()
        else:
            G = initial_point.copy()
        
        velocity = np.zeros_like(G)
        value_history = []
        grad_norm_history = []
        
        converged = False
        
        for iteration in range(self.max_iter):
            # Compute objective and gradient
            value = objective(G)
            euclidean_grad = gradient(G)
            
            # For GL(D), Riemannian gradient = Euclidean gradient (flat metric)
            # For O(D), would need to project to tangent space
            riemannian_grad = euclidean_grad
            
            grad_norm = np.linalg.norm(riemannian_grad)
            
            value_history.append(value)
            grad_norm_history.append(grad_norm)
            
            if verbose and iteration % 100 == 0:
                print(f"Iter {iteration}: value={value:.6f}, grad_norm={grad_norm:.6f}")
            
            # Check convergence
            if grad_norm < self.tol:
                converged = True
                break
            
            # Update with momentum
            velocity = self.momentum * velocity - self.lr * riemannian_grad
            
            # Retraction (project back to manifold)
            G = self._retraction(G, velocity)
        
        return RiemannianOptimizationResult(
            optimal_point=G,
            optimal_value=objective(G),
            gradient_norm_history=grad_norm_history,
            value_history=value_history,
            iterations=iteration + 1,
            converged=converged
        )
    
    def _retraction(self, G: np.ndarray, V: np.ndarray) -> np.ndarray:
        """
        Retraction map: move from G in direction V and project back to manifold.
        
        For GL(D): R_G(V) = G + V (with invertibility check)
        For O(D): R_G(V) = qf(G + V) where qf is QR-based retraction
        """
        G_new = G + V
        return self.manifold.project_to_manifold(G_new)


class AdaptiveRiemannianOptimizer:
    """
    Adaptive Riemannian optimizer with line search.
    
    Implements Armijo backtracking line search for adaptive step sizes.
    """
    
    def __init__(self, manifold: GLManifold, initial_lr: float = 1.0,
                 max_iterations: int = 1000, tolerance: float = 1e-6,
                 armijo_c: float = 1e-4, armijo_beta: float = 0.5):
        self.manifold = manifold
        self.initial_lr = initial_lr
        self.max_iter = max_iterations
        self.tol = tolerance
        self.armijo_c = armijo_c
        self.armijo_beta = armijo_beta
    
    def optimize(self, objective: Callable[[np.ndarray], float],
                 gradient: Callable[[np.ndarray], np.ndarray],
                 initial_point: Optional[np.ndarray] = None) -> RiemannianOptimizationResult:
        """Optimize with adaptive step sizes."""
        if initial_point is None:
            G = self.manifold.random_point()
        else:
            G = initial_point.copy()
        
        value_history = []
        grad_norm_history = []
        
        for iteration in range(self.max_iter):
            value = objective(G)
            grad = gradient(G)
            grad_norm = np.linalg.norm(grad)
            
            value_history.append(value)
            grad_norm_history.append(grad_norm)
            
            if grad_norm < self.tol:
                return RiemannianOptimizationResult(
                    optimal_point=G, optimal_value=value,
                    gradient_norm_history=grad_norm_history,
                    value_history=value_history,
                    iterations=iteration + 1, converged=True
                )
            
            # Armijo line search
            lr = self.initial_lr
            direction = -grad
            
            for _ in range(20):  # Max line search iterations
                G_new = self.manifold.project_to_manifold(G + lr * direction)
                new_value = objective(G_new)
                
                # Armijo condition
                if new_value <= value + self.armijo_c * lr * np.sum(grad * direction):
                    break
                lr *= self.armijo_beta
            
            G = G_new
        
        return RiemannianOptimizationResult(
            optimal_point=G, optimal_value=objective(G),
            gradient_norm_history=grad_norm_history,
            value_history=value_history,
            iterations=self.max_iter, converged=False
        )


def numerical_gradient(objective: Callable[[np.ndarray], float],
                       G: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
    """Compute numerical gradient using finite differences."""
    grad = np.zeros_like(G)
    for i in range(G.shape[0]):
        for j in range(G.shape[1]):
            G_plus = G.copy()
            G_plus[i, j] += epsilon
            G_minus = G.copy()
            G_minus[i, j] -= epsilon
            grad[i, j] = (objective(G_plus) - objective(G_minus)) / (2 * epsilon)
    return grad


if __name__ == "__main__":
    print("Riemannian Optimization Module Test")
    
    # Test on simple quadratic objective on GL(2)
    manifold = GLManifold(dimension=2)
    
    target = np.array([[1.0, 0.5], [0.2, 1.0]])
    
    def objective(G):
        return np.sum((G - target) ** 2)
    
    def gradient(G):
        return 2 * (G - target)
    
    optimizer = RiemannianGradientDescent(
        manifold, learning_rate=0.1, max_iterations=500
    )
    
    result = optimizer.optimize(objective, gradient, verbose=True)
    
    print(f"\nOptimization completed:")
    print(f"Converged: {result.converged}")
    print(f"Iterations: {result.iterations}")
    print(f"Final value: {result.optimal_value:.6f}")
    print(f"Target:\n{target}")
    print(f"Result:\n{result.optimal_point}")
