"""Tests for gauge optimization module."""

import numpy as np
import sys
sys.path.insert(0, '..')
from src.positivity_gauge_opt import (
    PositivityGaugeOptimizer, 
    PositivityPhase,
    create_random_mps,
    create_ghz_mps
)


def test_optimizer_convergence():
    """Test that optimizer converges."""
    np.random.seed(42)
    mps = create_random_mps(num_sites=4, bond_dim=2)
    optimizer = PositivityGaugeOptimizer(max_iterations=50)
    result = optimizer.optimize(mps.tensors)
    assert len(result.convergence_history) > 0
    assert result.iterations > 0


def test_positivity_score_bounds():
    """Test positivity score is in valid range."""
    np.random.seed(42)
    mps = create_random_mps(num_sites=4, bond_dim=2)
    optimizer = PositivityGaugeOptimizer(max_iterations=20)
    result = optimizer.optimize(mps.tensors)
    assert 0 <= result.positivity_score <= 1


def test_phase_classification():
    """Test phase classification."""
    ghz = create_ghz_mps(num_sites=4)
    optimizer = PositivityGaugeOptimizer(max_iterations=50)
    result = optimizer.optimize(ghz.tensors)
    assert result.phase in list(PositivityPhase)


def test_trivial_case():
    """Test single tensor case."""
    tensor = np.random.randn(1, 2, 1)
    optimizer = PositivityGaugeOptimizer()
    result = optimizer.optimize([tensor])
    assert result.converged


if __name__ == "__main__":
    test_optimizer_convergence()
    test_positivity_score_bounds()
    test_phase_classification()
    test_trivial_case()
    print("All gauge optimizer tests passed!")
