"""
Feature Extraction for QUANTUM-PHASE-GNN
========================================

Utilities for extracting structural and algebraic features from quantum circuits.
"""

import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass

from .dla import DLAComputer, extract_circuit_generators


@dataclass
class CircuitFeatures:
    """Complete feature set for a quantum circuit."""
    dla_dimension: int
    dla_ratio: float
    dla_log_normalized: float
    num_qubits: int
    num_gates: int
    circuit_depth: int
    single_qubit_ratio: float
    two_qubit_ratio: float
    magic_gate_ratio: float
    entanglement_density: float
    max_entanglement_distance: int
    average_entanglement_distance: float
    qubit_connectivity: float
    gate_locality: float
    
    def to_vector(self) -> np.ndarray:
        """Convert to feature vector."""
        return np.array([
            self.dla_ratio, self.dla_log_normalized,
            self.num_qubits / 20, self.num_gates / 100, self.circuit_depth / 50,
            self.single_qubit_ratio, self.two_qubit_ratio, self.magic_gate_ratio,
            self.entanglement_density,
            self.max_entanglement_distance / max(self.num_qubits, 1),
            self.average_entanglement_distance / max(self.num_qubits - 1, 1),
            self.qubit_connectivity, self.gate_locality
        ])


class FeatureExtractor:
    """Extract features from quantum circuits for GNN classification."""
    
    def __init__(self, max_dla_qubits: int = 8):
        self.max_dla_qubits = max_dla_qubits
    
    def extract(self, circuit: List[dict], n_qubits: int) -> CircuitFeatures:
        """Extract complete feature set from circuit."""
        dla_dim, dla_ratio, dla_log = self._extract_dla_features(circuit, n_qubits)
        num_gates = len(circuit)
        depth = self._compute_depth(circuit, n_qubits)
        single_q, two_q, magic = self._extract_gate_composition(circuit)
        ent_density, max_dist, avg_dist = self._extract_entanglement_features(circuit, n_qubits)
        connectivity = self._compute_qubit_connectivity(circuit, n_qubits)
        locality = self._compute_gate_locality(circuit, n_qubits)
        
        return CircuitFeatures(
            dla_dimension=dla_dim, dla_ratio=dla_ratio, dla_log_normalized=dla_log,
            num_qubits=n_qubits, num_gates=num_gates, circuit_depth=depth,
            single_qubit_ratio=single_q, two_qubit_ratio=two_q, magic_gate_ratio=magic,
            entanglement_density=ent_density, max_entanglement_distance=max_dist,
            average_entanglement_distance=avg_dist, qubit_connectivity=connectivity,
            gate_locality=locality
        )
    
    def _extract_dla_features(self, circuit: List[dict], n_qubits: int) -> Tuple[int, float, float]:
        if n_qubits <= self.max_dla_qubits:
            generators = extract_circuit_generators(circuit, n_qubits)
            if len(generators) > 0:
                dla = DLAComputer(n_qubits)
                dim = dla.compute_dimension(generators)
            else:
                dim = 0
        else:
            dim = self._estimate_dla_dimension(circuit, n_qubits)
        max_dim = 4 ** n_qubits - 1
        ratio = dim / max_dim if max_dim > 0 else 0.0
        log_norm = np.log(dim + 1) / np.log(max_dim + 1) if max_dim > 0 else 0.0
        return dim, ratio, log_norm
    
    def _estimate_dla_dimension(self, circuit: List[dict], n_qubits: int) -> int:
        single_q_gens = set()
        two_q_pairs = set()
        for gate in circuit:
            qubits = tuple(sorted(gate['qubits']))
            if len(qubits) == 1:
                single_q_gens.add((qubits[0], gate['gate'][0] if gate['gate'] else 'R'))
            elif len(qubits) == 2:
                two_q_pairs.add(qubits)
        local_gens = len(single_q_gens) * 3
        interaction_gens = len(two_q_pairs) * 15
        return min(local_gens + interaction_gens, 4 ** n_qubits - 1)
    
    def _compute_depth(self, circuit: List[dict], n_qubits: int) -> int:
        qubit_depths = [0] * n_qubits
        for gate in circuit:
            qubits = gate['qubits']
            max_depth = max(qubit_depths[q] for q in qubits) if qubits else 0
            new_depth = max_depth + 1
            for q in qubits:
                qubit_depths[q] = new_depth
        return max(qubit_depths) if qubit_depths else 0
    
    def _extract_gate_composition(self, circuit: List[dict]) -> Tuple[float, float, float]:
        if not circuit:
            return 0.0, 0.0, 0.0
        single_q = sum(1 for g in circuit if len(g['qubits']) == 1)
        two_q = sum(1 for g in circuit if len(g['qubits']) == 2)
        magic_gates = {'T', 'Tdg', 'S', 'Sdg'}
        magic = sum(1 for g in circuit if g['gate'] in magic_gates)
        total = len(circuit)
        return single_q / total, two_q / total, magic / total
    
    def _extract_entanglement_features(self, circuit: List[dict], n_qubits: int) -> Tuple[float, int, float]:
        distances = []
        pairs = set()
        for gate in circuit:
            if len(gate['qubits']) == 2:
                q1, q2 = gate['qubits']
                distances.append(abs(q1 - q2))
                pairs.add(tuple(sorted([q1, q2])))
        max_pairs = n_qubits * (n_qubits - 1) / 2 if n_qubits > 1 else 1
        density = len(pairs) / max_pairs
        max_dist = max(distances) if distances else 0
        avg_dist = np.mean(distances) if distances else 0.0
        return density, max_dist, avg_dist
    
    def _compute_qubit_connectivity(self, circuit: List[dict], n_qubits: int) -> float:
        pairs = set()
        for gate in circuit:
            if len(gate['qubits']) == 2:
                pairs.add(tuple(sorted(gate['qubits'])))
        max_pairs = n_qubits * (n_qubits - 1) / 2 if n_qubits > 1 else 1
        return len(pairs) / max_pairs
    
    def _compute_gate_locality(self, circuit: List[dict], n_qubits: int) -> float:
        distances = []
        for gate in circuit:
            if len(gate['qubits']) == 2:
                distances.append(abs(gate['qubits'][0] - gate['qubits'][1]))
        if not distances or n_qubits <= 1:
            return 1.0
        avg_dist = np.mean(distances)
        max_dist = n_qubits - 1
        return 1.0 - (avg_dist - 1) / (max_dist - 1) if max_dist > 1 else 1.0


def extract_boundary_features(circuit: List[dict], n_qubits: int) -> Dict[str, float]:
    """Extract features for boundary characterization."""
    extractor = FeatureExtractor()
    features = extractor.extract(circuit, n_qubits)
    return {
        'dla_ratio': features.dla_ratio,
        'entanglement_density': features.entanglement_density,
        'magic_content': features.magic_gate_ratio,
        'locality': features.gate_locality,
        'trainability_proxy': 1.0 - features.dla_ratio,
        'simulability_proxy': 1.0 - features.entanglement_density * features.magic_gate_ratio,
        'boundary_score': abs(features.dla_ratio - 0.5)
    }
