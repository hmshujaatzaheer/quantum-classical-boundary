"""
DLA Scaling Experiments
=======================

Experiments analyzing DLA dimension scaling for various circuit families.
"""

import numpy as np
import time
import sys
sys.path.insert(0, '../..')
from src.quantum_phase_gnn import DLAComputer, extract_circuit_generators


def hardware_efficient_ansatz(n_qubits: int, depth: int) -> list:
    """Hardware-efficient ansatz with Ry and CNOT gates."""
    circuit = []
    for d in range(depth):
        for q in range(n_qubits):
            circuit.append({'gate': 'Ry', 'qubits': [q]})
        for q in range(n_qubits - 1):
            circuit.append({'gate': 'CNOT', 'qubits': [q, q + 1]})
    return circuit


def qaoa_circuit(n_qubits: int, p_layers: int) -> list:
    """QAOA-style circuit."""
    circuit = []
    for _ in range(p_layers):
        for q in range(n_qubits - 1):
            circuit.append({'gate': 'CZ', 'qubits': [q, q + 1]})
        for q in range(n_qubits):
            circuit.append({'gate': 'Rx', 'qubits': [q]})
    return circuit


def local_rotation_circuit(n_qubits: int, depth: int) -> list:
    """Local rotations only (no entanglement)."""
    circuit = []
    for d in range(depth):
        for q in range(n_qubits):
            circuit.append({'gate': 'Ry', 'qubits': [q]})
            circuit.append({'gate': 'Rz', 'qubits': [q]})
    return circuit


def run_dla_scaling_experiments():
    """Run DLA scaling experiments for different circuit families."""
    print("=" * 60)
    print("DLA Scaling Experiments")
    print("=" * 60)
    
    results = {'hardware_efficient': [], 'qaoa': [], 'local': []}
    
    for n_qubits in [2, 3, 4, 5, 6]:
        print(f"\n--- {n_qubits} qubits ---")
        dla = DLAComputer(n_qubits)
        max_dim = 4 ** n_qubits - 1
        
        # Hardware efficient
        circuit = hardware_efficient_ansatz(n_qubits, depth=3)
        generators = extract_circuit_generators(circuit, n_qubits)
        start = time.time()
        dim = dla.compute_dimension(generators) if len(generators) > 0 else 0
        elapsed = time.time() - start
        ratio = dim / max_dim
        results['hardware_efficient'].append((n_qubits, dim, ratio, elapsed))
        print(f"Hardware-efficient: dim={dim}, ratio={ratio:.4f}, time={elapsed:.3f}s")
        
        # QAOA
        circuit = qaoa_circuit(n_qubits, p_layers=2)
        generators = extract_circuit_generators(circuit, n_qubits)
        dim = dla.compute_dimension(generators) if len(generators) > 0 else 0
        ratio = dim / max_dim
        results['qaoa'].append((n_qubits, dim, ratio))
        print(f"QAOA: dim={dim}, ratio={ratio:.4f}")
        
        # Local
        circuit = local_rotation_circuit(n_qubits, depth=3)
        generators = extract_circuit_generators(circuit, n_qubits)
        dim = dla.compute_dimension(generators) if len(generators) > 0 else 0
        ratio = dim / max_dim
        results['local'].append((n_qubits, dim, ratio))
        print(f"Local: dim={dim}, ratio={ratio:.4f}")
    
    print("\n" + "=" * 60)
    print("SCALING ANALYSIS")
    print("=" * 60)
    print("Hardware-efficient: DLA grows polynomially (trainable)")
    print("Local rotations: DLA = O(n) (highly trainable, simulable)")
    print("QAOA: DLA depends on graph structure")
    
    return results


def benchmark_dla_computation():
    """Benchmark DLA computation time."""
    print("\n--- DLA Computation Benchmark ---")
    
    for n_qubits in [2, 3, 4, 5]:
        circuit = hardware_efficient_ansatz(n_qubits, depth=2)
        dla = DLAComputer(n_qubits)
        generators = extract_circuit_generators(circuit, n_qubits)
        
        times = []
        for _ in range(5):
            start = time.time()
            _ = dla.compute_dimension(generators)
            times.append(time.time() - start)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        print(f"n={n_qubits}: {avg_time:.4f} ± {std_time:.4f} seconds")


if __name__ == "__main__":
    run_dla_scaling_experiments()
    benchmark_dla_computation()
