"""
QUANTUM-PHASE-GNN: Graph Neural Network for Quantum Phase Classification
=========================================================================

Implementation of Algorithm 2: QUANTUM-PHASE-GNN from the PhD proposal.

This GNN classifies quantum circuits into computational phases:
- TRAINABLE: Small DLA, no barren plateau
- TRANSITION: Boundary region with critical behavior  
- BARREN PLATEAU: Large DLA, exponentially vanishing gradients

Author: PhD Research - ETH Zurich Quantum Computing Group
"""

import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum

from .dla import DLAComputer, extract_circuit_generators


class QuantumPhase(Enum):
    """Quantum computational phases."""
    TRAINABLE = 0
    TRANSITION = 1
    BARREN_PLATEAU = 2


@dataclass
class CircuitGraph:
    """Graph representation of quantum circuit for GNN input."""
    num_qubits: int
    node_features: np.ndarray
    edge_index: np.ndarray
    edge_features: np.ndarray
    global_features: np.ndarray


@dataclass
class PhaseClassification:
    """Result of phase classification."""
    predicted_phase: QuantumPhase
    confidence: float
    dla_ratio: float
    trainability_score: float
    phase_probabilities: Dict[QuantumPhase, float]


class QuantumPhaseGNN:
    """
    Graph Neural Network for quantum phase classification.
    
    ALGORITHM 2: QUANTUM-PHASE-GNN
    ==============================
    
    Input: Quantum circuit C
    Output: Phase classification (TRAINABLE/TRANSITION/BARREN_PLATEAU)
    
    Procedure:
    1. EXTRACT FEATURES:
       a. Compute DLA dimension and ratio r_DLA
       b. Extract circuit graph structure
       c. Compute magic depth and entanglement features
       
    2. GRAPH CONSTRUCTION:
       a. Create node features for qubits and gates
       b. Build edge connections (qubit-gate, temporal)
       c. Add global DLA features
       
    3. GNN FORWARD PASS:
       a. Message passing on circuit graph
       b. Aggregate node embeddings
       c. Combine with global features
       
    4. CLASSIFICATION:
       a. MLP classifier on combined features
       b. Output phase probabilities
       c. Return predicted phase with confidence
    """
    
    def __init__(self, hidden_dim: int = 64, num_layers: int = 3, dropout: float = 0.1):
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.node_feature_dim = 16
        self.edge_feature_dim = 8
        self.global_feature_dim = 12
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        np.random.seed(42)
        self.W_node = np.random.randn(self.node_feature_dim, self.hidden_dim) * 0.1
        self.W_msg = [np.random.randn(self.hidden_dim, self.hidden_dim) * 0.1 
                      for _ in range(self.num_layers)]
        self.W_update = [np.random.randn(2 * self.hidden_dim, self.hidden_dim) * 0.1
                         for _ in range(self.num_layers)]
        self.W_global = np.random.randn(self.global_feature_dim, self.hidden_dim) * 0.1
        self.W_classifier = np.random.randn(2 * self.hidden_dim, 3) * 0.1
    
    def classify(self, circuit: List[dict], n_qubits: int) -> PhaseClassification:
        """Classify quantum circuit into computational phase."""
        graph = self._build_circuit_graph(circuit, n_qubits)
        node_embeddings = self._gnn_forward(graph)
        logits = self._classify(node_embeddings, graph.global_features)
        probs = self._softmax(logits)
        
        phase_idx = np.argmax(probs)
        phase = list(QuantumPhase)[phase_idx]
        confidence = probs[phase_idx]
        trainability = 1.0 - graph.global_features[0]
        
        return PhaseClassification(
            predicted_phase=phase,
            confidence=float(confidence),
            dla_ratio=float(graph.global_features[0]),
            trainability_score=float(trainability),
            phase_probabilities={
                QuantumPhase.TRAINABLE: float(probs[0]),
                QuantumPhase.TRANSITION: float(probs[1]),
                QuantumPhase.BARREN_PLATEAU: float(probs[2])
            }
        )
    
    def _build_circuit_graph(self, circuit: List[dict], n_qubits: int) -> CircuitGraph:
        """Build graph representation of quantum circuit."""
        generators = extract_circuit_generators(circuit, n_qubits)
        dla = DLAComputer(n_qubits)
        
        if len(generators) > 0:
            dla_dim = dla.compute_dimension(generators)
            dla_ratio = dla_dim / (4 ** n_qubits - 1)
        else:
            dla_dim = 0
            dla_ratio = 0.0
        
        num_gates = len(circuit)
        num_nodes = n_qubits + num_gates
        node_features = np.zeros((num_nodes, self.node_feature_dim))
        
        for q in range(n_qubits):
            node_features[q, 0] = 1.0
            node_features[q, 1] = q / n_qubits
        
        for g, gate_info in enumerate(circuit):
            node_idx = n_qubits + g
            node_features[node_idx, 2] = 1.0
            node_features[node_idx, 3:7] = self._encode_gate_type(gate_info['gate'])
            node_features[node_idx, 7] = len(gate_info['qubits']) / 2
        
        edges = []
        edge_features_list = []
        
        for g, gate_info in enumerate(circuit):
            gate_node = n_qubits + g
            for qubit in gate_info['qubits']:
                edges.append([qubit, gate_node])
                edge_features_list.append([1.0, 0.0, g / max(num_gates, 1), 0.0, 0.0, 0.0, 0.0, 0.0])
                edges.append([gate_node, qubit])
                edge_features_list.append([0.0, 1.0, g / max(num_gates, 1), 0.0, 0.0, 0.0, 0.0, 0.0])
        
        for g in range(num_gates - 1):
            edges.append([n_qubits + g, n_qubits + g + 1])
            edge_features_list.append([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
        
        edge_index = np.array(edges).T if edges else np.zeros((2, 0))
        edge_features = np.array(edge_features_list) if edge_features_list else np.zeros((0, self.edge_feature_dim))
        
        global_features = np.array([
            dla_ratio,
            np.log(dla_dim + 1) / np.log(4 ** n_qubits) if n_qubits > 0 else 0,
            self._compute_magic_depth(circuit),
            self._compute_entanglement_density(circuit, n_qubits),
            n_qubits / 20,
            num_gates / (n_qubits * 10) if n_qubits > 0 else 0,
            self._count_two_qubit_gates(circuit) / max(num_gates, 1),
            self._compute_locality(circuit, n_qubits),
            0.0, 0.0, 0.0, 0.0
        ])
        
        return CircuitGraph(
            num_qubits=n_qubits,
            node_features=node_features,
            edge_index=edge_index,
            edge_features=edge_features,
            global_features=global_features
        )
    
    def _gnn_forward(self, graph: CircuitGraph) -> np.ndarray:
        """GNN forward pass with message passing."""
        h = np.tanh(graph.node_features @ self.W_node)
        for layer in range(self.num_layers):
            messages = np.zeros_like(h)
            if graph.edge_index.shape[1] > 0:
                for e in range(graph.edge_index.shape[1]):
                    src, dst = graph.edge_index[:, e]
                    msg = h[src] @ self.W_msg[layer]
                    messages[dst] += msg
            combined = np.concatenate([h, messages], axis=1)
            h = np.tanh(combined @ self.W_update[layer])
        return h
    
    def _classify(self, node_embeddings: np.ndarray, global_features: np.ndarray) -> np.ndarray:
        """Final classification from embeddings."""
        graph_embedding = np.mean(node_embeddings, axis=0)
        global_embedding = np.tanh(global_features @ self.W_global)
        combined = np.concatenate([graph_embedding, global_embedding])
        return combined @ self.W_classifier
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
    
    def _encode_gate_type(self, gate: str) -> np.ndarray:
        encoding = np.zeros(4)
        gate_types = {'Rx': 0, 'Ry': 0, 'Rz': 0, 'X': 0, 'Y': 0, 'Z': 0,
                      'CNOT': 1, 'CX': 1, 'CZ': 2, 'H': 3, 'T': 3, 'S': 3}
        idx = gate_types.get(gate, 0)
        encoding[idx] = 1.0
        return encoding
    
    def _compute_magic_depth(self, circuit: List[dict]) -> float:
        magic_gates = {'T', 'Tdg', 'S', 'Sdg'}
        magic_count = sum(1 for g in circuit if g['gate'] in magic_gates)
        return magic_count / max(len(circuit), 1)
    
    def _compute_entanglement_density(self, circuit: List[dict], n_qubits: int) -> float:
        two_qubit_gates = sum(1 for g in circuit if len(g['qubits']) == 2)
        max_entangling = n_qubits * (n_qubits - 1) / 2 if n_qubits > 1 else 1
        return two_qubit_gates / max(max_entangling, 1)
    
    def _count_two_qubit_gates(self, circuit: List[dict]) -> int:
        return sum(1 for g in circuit if len(g['qubits']) == 2)
    
    def _compute_locality(self, circuit: List[dict], n_qubits: int) -> float:
        distances = []
        for g in circuit:
            if len(g['qubits']) == 2:
                distances.append(abs(g['qubits'][0] - g['qubits'][1]))
        if distances and n_qubits > 1:
            return 1.0 - np.mean(distances) / (n_qubits - 1)
        return 1.0


if __name__ == "__main__":
    print("QUANTUM-PHASE-GNN - Algorithm 2 Implementation")
    gnn = QuantumPhaseGNN()
    circuit = [
        {'gate': 'Ry', 'qubits': [0]},
        {'gate': 'Ry', 'qubits': [1]},
        {'gate': 'CNOT', 'qubits': [0, 1]},
    ]
    result = gnn.classify(circuit, n_qubits=2)
    print(f"Phase: {result.predicted_phase.name}")
    print(f"Confidence: {result.confidence:.4f}")
    print(f"DLA ratio: {result.dla_ratio:.4f}")
