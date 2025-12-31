"""Tests for DLA computation module."""

import numpy as np
import sys
sys.path.insert(0, '..')
from src.quantum_phase_gnn import DLAComputer, extract_circuit_generators


def test_dla_single_qubit():
    """Test DLA for single-qubit rotations."""
    dla = DLAComputer(n_qubits=1)
    circuit = [{'gate': 'Ry', 'qubits': [0]}]
    generators = extract_circuit_generators(circuit, n_qubits=1)
    dim = dla.compute_dimension(generators)
    assert dim >= 1, "Single rotation should have DLA dim >= 1"


def test_dla_two_qubit():
    """Test DLA for two-qubit circuit."""
    dla = DLAComputer(n_qubits=2)
    circuit = [
        {'gate': 'Ry', 'qubits': [0]},
        {'gate': 'Ry', 'qubits': [1]},
        {'gate': 'CNOT', 'qubits': [0, 1]},
    ]
    generators = extract_circuit_generators(circuit, n_qubits=2)
    dim = dla.compute_dimension(generators)
    assert dim <= 15, "DLA dimension should not exceed 4^n - 1"


def test_dla_ratio():
    """Test DLA ratio computation."""
    dla = DLAComputer(n_qubits=2)
    circuit = [
        {'gate': 'Ry', 'qubits': [0]},
        {'gate': 'CNOT', 'qubits': [0, 1]},
    ]
    generators = extract_circuit_generators(circuit, n_qubits=2)
    ratio = dla.compute_dla_ratio(generators)
    assert 0 <= ratio <= 1, "DLA ratio should be in [0, 1]"


def test_empty_circuit():
    """Test empty circuit."""
    dla = DLAComputer(n_qubits=2)
    dim = dla.compute_dimension([])
    assert dim == 0, "Empty circuit should have DLA dim 0"


if __name__ == "__main__":
    test_dla_single_qubit()
    test_dla_two_qubit()
    test_dla_ratio()
    test_empty_circuit()
    print("All DLA tests passed!")
