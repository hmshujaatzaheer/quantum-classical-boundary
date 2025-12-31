"""
QUANTUM-PHASE-GNN Package
=========================

Graph Neural Network framework for quantum phase classification.
Implements Research Thrust 1: Boundary Characterization.
"""

from .dla import DLAComputer, extract_circuit_generators
from .model import QuantumPhaseGNN, QuantumPhase, PhaseClassification, CircuitGraph
from .features import FeatureExtractor, CircuitFeatures, extract_boundary_features
from .states import (computational_basis, zero_state, plus_state, ghz_state, w_state,
                     random_state, random_product_state, fidelity, entanglement_entropy)

__all__ = [
    'DLAComputer', 'extract_circuit_generators',
    'QuantumPhaseGNN', 'QuantumPhase', 'PhaseClassification', 'CircuitGraph',
    'FeatureExtractor', 'CircuitFeatures', 'extract_boundary_features',
    'computational_basis', 'zero_state', 'plus_state', 'ghz_state', 'w_state',
    'random_state', 'random_product_state', 'fidelity', 'entanglement_entropy',
]

__version__ = '0.1.0'
