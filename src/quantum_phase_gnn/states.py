"""
Quantum State Utilities
=======================

Utilities for creating and manipulating quantum states.
"""

import numpy as np
from typing import List, Optional


def computational_basis(n_qubits: int, state: int) -> np.ndarray:
    """Create computational basis state |state⟩."""
    dim = 2 ** n_qubits
    psi = np.zeros(dim, dtype=complex)
    psi[state] = 1.0
    return psi


def zero_state(n_qubits: int) -> np.ndarray:
    """Create |00...0⟩ state."""
    return computational_basis(n_qubits, 0)


def plus_state(n_qubits: int) -> np.ndarray:
    """Create |+⟩^⊗n state (equal superposition)."""
    dim = 2 ** n_qubits
    return np.ones(dim, dtype=complex) / np.sqrt(dim)


def ghz_state(n_qubits: int) -> np.ndarray:
    """Create GHZ state: (|00...0⟩ + |11...1⟩) / √2"""
    psi = np.zeros(2 ** n_qubits, dtype=complex)
    psi[0] = 1.0 / np.sqrt(2)
    psi[-1] = 1.0 / np.sqrt(2)
    return psi


def w_state(n_qubits: int) -> np.ndarray:
    """Create W state: (|10...0⟩ + |01...0⟩ + ... + |00...1⟩) / √n"""
    dim = 2 ** n_qubits
    psi = np.zeros(dim, dtype=complex)
    for i in range(n_qubits):
        idx = 1 << (n_qubits - 1 - i)
        psi[idx] = 1.0 / np.sqrt(n_qubits)
    return psi


def random_state(n_qubits: int, seed: Optional[int] = None) -> np.ndarray:
    """Create random (Haar-distributed) pure state."""
    if seed is not None:
        np.random.seed(seed)
    dim = 2 ** n_qubits
    psi = np.random.randn(dim) + 1j * np.random.randn(dim)
    return psi / np.linalg.norm(psi)


def random_product_state(n_qubits: int, seed: Optional[int] = None) -> np.ndarray:
    """Create random product state (no entanglement)."""
    if seed is not None:
        np.random.seed(seed)
    theta = np.random.uniform(0, np.pi)
    phi = np.random.uniform(0, 2 * np.pi)
    single = np.array([np.cos(theta / 2), np.exp(1j * phi) * np.sin(theta / 2)])
    psi = single
    for _ in range(n_qubits - 1):
        theta = np.random.uniform(0, np.pi)
        phi = np.random.uniform(0, 2 * np.pi)
        single = np.array([np.cos(theta / 2), np.exp(1j * phi) * np.sin(theta / 2)])
        psi = np.kron(psi, single)
    return psi


def pure_to_density(psi: np.ndarray) -> np.ndarray:
    """Convert pure state to density matrix: ρ = |ψ⟩⟨ψ|"""
    return np.outer(psi, np.conj(psi))


def fidelity(psi1: np.ndarray, psi2: np.ndarray) -> float:
    """Compute fidelity between two pure states: F = |⟨ψ₁|ψ₂⟩|²"""
    return np.abs(np.vdot(psi1, psi2)) ** 2


def entanglement_entropy(psi: np.ndarray, n_qubits: int, subsystem_A: List[int]) -> float:
    """Compute entanglement entropy of subsystem A."""
    shape = [2] * n_qubits
    tensor = psi.reshape(shape)
    all_qubits = list(range(n_qubits))
    subsystem_B = [q for q in all_qubits if q not in subsystem_A]
    order = subsystem_A + subsystem_B
    tensor = np.transpose(tensor, order)
    dim_A = 2 ** len(subsystem_A)
    dim_B = 2 ** len(subsystem_B)
    matrix = tensor.reshape(dim_A, dim_B)
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    schmidt_coeffs = singular_values ** 2
    schmidt_coeffs = schmidt_coeffs[schmidt_coeffs > 1e-15]
    return -np.sum(schmidt_coeffs * np.log2(schmidt_coeffs))


if __name__ == "__main__":
    print("Quantum State Utilities Test")
    n = 3
    psi_ghz = ghz_state(n)
    print(f"GHZ state norm: {np.linalg.norm(psi_ghz):.6f}")
    print(f"Entanglement entropy: {entanglement_entropy(psi_ghz, n, [0]):.6f}")
