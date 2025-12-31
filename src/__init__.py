"""
Quantum-Classical Boundary Research Framework
==============================================

A comprehensive framework for characterizing quantum-classical computational boundaries
using machine learning approaches to trainability, simulability, and verification.

Three Research Thrusts:
1. QUANTUM-PHASE-GNN: ML-based boundary classification
2. POSITIVITY-GAUGE-OPT: GNN-guided tensor network optimization  
3. BELL-VERIFY-NEAR-TERM: Practical verification protocols

Author: PhD Research - ETH Zurich Quantum Computing Group
Repository: https://github.com/hmshujaatzaheer/quantum-classical-boundary
"""

from .quantum_phase_gnn import (
    DLAComputer, QuantumPhaseGNN, QuantumPhase, FeatureExtractor
)
from .positivity_gauge_opt import (
    PositivityGaugeOptimizer, PositivityPhase, TensorNetwork
)
from .bell_verify import (
    BellVerificationProtocol, VerificationOutcome, DeviceCharacteristics
)

__all__ = [
    'DLAComputer', 'QuantumPhaseGNN', 'QuantumPhase', 'FeatureExtractor',
    'PositivityGaugeOptimizer', 'PositivityPhase', 'TensorNetwork',
    'BellVerificationProtocol', 'VerificationOutcome', 'DeviceCharacteristics'
]

__version__ = '0.1.0'
__author__ = 'PhD Research - ETH Zurich'
