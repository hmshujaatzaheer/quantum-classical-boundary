"""
POSITIVITY-GAUGE-OPT Package
============================

Gauge optimization for sign-problem-free tensor networks.
Implements Research Thrust 2: GNN-Guided Tensor Network Optimization.

Includes cotengra integration for efficient tensor contraction.
"""

from .gauge_optimizer import (
    PositivityGaugeOptimizer,
    PositivityPhase,
    GaugeOptimizationResult,
    TensorNetworkConfig,
    verify_approximation_guarantee
)
from .tensor_network import (
    TensorNetwork,
    create_random_mps,
    create_product_mps,
    create_ghz_mps,
    contract_mps,
    normalize_mps,
    left_canonicalize,
    compute_bond_entropy,
    analyze_positivity
)
from .riemannian import (
    GLManifold,
    OrthogonalManifold,
    RiemannianGradientDescent,
    AdaptiveRiemannianOptimizer,
    RiemannianOptimizationResult,
    numerical_gradient
)

__all__ = [
    'PositivityGaugeOptimizer', 'PositivityPhase', 'GaugeOptimizationResult',
    'TensorNetworkConfig', 'verify_approximation_guarantee',
    'TensorNetwork', 'create_random_mps', 'create_product_mps', 'create_ghz_mps',
    'contract_mps', 'normalize_mps', 'left_canonicalize',
    'compute_bond_entropy', 'analyze_positivity'
]

__version__ = '0.1.0'
