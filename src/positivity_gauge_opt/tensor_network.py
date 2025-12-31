"""
Tensor Network Utilities
========================

Utilities for creating and manipulating tensor networks.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class TensorNetwork:
    """Generic tensor network representation."""
    tensors: List[np.ndarray]
    structure: str
    bond_dims: List[int]
    phys_dim: int


def create_random_mps(num_sites: int, bond_dim: int, phys_dim: int = 2,
                      seed: Optional[int] = None) -> TensorNetwork:
    """Create random Matrix Product State."""
    if seed is not None:
        np.random.seed(seed)
    tensors = []
    bond_dims = []
    for i in range(num_sites):
        D_left = 1 if i == 0 else min(bond_dim, phys_dim ** i)
        D_right = 1 if i == num_sites - 1 else min(bond_dim, phys_dim ** (num_sites - i - 1))
        T = np.random.randn(D_left, phys_dim, D_right) + 1j * np.random.randn(D_left, phys_dim, D_right)
        T /= np.linalg.norm(T)
        tensors.append(T)
        if i < num_sites - 1:
            bond_dims.append(D_right)
    return TensorNetwork(tensors=tensors, structure='mps', bond_dims=bond_dims, phys_dim=phys_dim)


def create_product_mps(states: List[np.ndarray]) -> TensorNetwork:
    """Create MPS representing product state."""
    tensors = [state.reshape(1, len(state), 1) for state in states]
    return TensorNetwork(tensors=tensors, structure='mps', bond_dims=[1] * (len(states) - 1), phys_dim=len(states[0]))


def create_ghz_mps(num_sites: int) -> TensorNetwork:
    """Create MPS for GHZ state: (|00...0⟩ + |11...1⟩) / √2"""
    tensors = []
    T0 = np.zeros((1, 2, 2), dtype=complex)
    T0[0, 0, 0] = 1.0 / np.sqrt(2)
    T0[0, 1, 1] = 1.0 / np.sqrt(2)
    tensors.append(T0)
    for _ in range(num_sites - 2):
        T = np.zeros((2, 2, 2), dtype=complex)
        T[0, 0, 0] = 1.0
        T[1, 1, 1] = 1.0
        tensors.append(T)
    TN = np.zeros((2, 2, 1), dtype=complex)
    TN[0, 0, 0] = 1.0
    TN[1, 1, 0] = 1.0
    tensors.append(TN)
    return TensorNetwork(tensors=tensors, structure='mps', bond_dims=[2] * (num_sites - 1), phys_dim=2)


def contract_mps(mps: TensorNetwork, config: Optional[List[int]] = None) -> complex:
    """Contract MPS to get amplitude or state vector."""
    if config is not None:
        sliced = [T[:, config[i], :] for i, T in enumerate(mps.tensors)]
        result = sliced[0]
        for i in range(1, len(sliced)):
            result = result @ sliced[i]
        return np.squeeze(result)
    else:
        dim = mps.phys_dim ** len(mps.tensors)
        state = np.zeros(dim, dtype=complex)
        for idx in range(dim):
            config = []
            temp = idx
            for _ in range(len(mps.tensors)):
                config.append(temp % mps.phys_dim)
                temp //= mps.phys_dim
            config = config[::-1]
            state[idx] = contract_mps(mps, config)
        return state


def normalize_mps(mps: TensorNetwork) -> TensorNetwork:
    """Normalize MPS to unit norm."""
    state = contract_mps(mps)
    norm = np.linalg.norm(state)
    tensors = [t.copy() for t in mps.tensors]
    tensors[0] /= norm
    return TensorNetwork(tensors=tensors, structure=mps.structure, bond_dims=mps.bond_dims, phys_dim=mps.phys_dim)


def left_canonicalize(mps: TensorNetwork) -> TensorNetwork:
    """Convert MPS to left-canonical form."""
    tensors = [t.copy() for t in mps.tensors]
    for i in range(len(tensors) - 1):
        T = tensors[i]
        D_left, d, D_right = T.shape
        M = T.reshape(D_left * d, D_right)
        Q, R = np.linalg.qr(M)
        tensors[i] = Q.reshape(D_left, d, Q.shape[1])
        tensors[i + 1] = np.tensordot(R, tensors[i + 1], axes=([1], [0]))
    return TensorNetwork(tensors=tensors, structure=mps.structure, bond_dims=mps.bond_dims, phys_dim=mps.phys_dim)


def compute_bond_entropy(mps: TensorNetwork, bond: int) -> float:
    """Compute entanglement entropy at given bond."""
    tensors = [t.copy() for t in mps.tensors]
    for i in range(bond):
        T = tensors[i]
        D_left, d, D_right = T.shape
        M = T.reshape(D_left * d, D_right)
        Q, R = np.linalg.qr(M)
        tensors[i] = Q.reshape(D_left, d, Q.shape[1])
        tensors[i + 1] = np.tensordot(R, tensors[i + 1], axes=([1], [0]))
    for i in range(len(tensors) - 1, bond, -1):
        T = tensors[i]
        D_left, d, D_right = T.shape
        M = T.reshape(D_left, d * D_right)
        Q, R = np.linalg.qr(M.T)
        tensors[i] = Q.T.reshape(Q.shape[1], d, D_right)
        tensors[i - 1] = np.tensordot(tensors[i - 1], R.T, axes=([-1], [0]))
    T = tensors[bond]
    D_left, d, D_right = T.shape
    M = T.reshape(D_left * d, D_right)
    _, S, _ = np.linalg.svd(M, full_matrices=False)
    spectrum = S ** 2
    spectrum = spectrum[spectrum > 1e-15]
    return -np.sum(spectrum * np.log2(spectrum))


def analyze_positivity(mps: TensorNetwork, num_samples: int = 1000) -> Dict:
    """Analyze positivity properties of MPS amplitudes."""
    amplitudes = []
    for _ in range(num_samples):
        config = [np.random.randint(0, mps.phys_dim) for _ in range(len(mps.tensors))]
        amp = contract_mps(mps, config)
        amplitudes.append(amp)
    amplitudes = np.array(amplitudes)
    real_parts = np.real(amplitudes)
    imag_parts = np.imag(amplitudes)
    positive = np.sum((real_parts > 0) & (np.abs(imag_parts) < 1e-10))
    negative = np.sum((real_parts < 0) & (np.abs(imag_parts) < 1e-10))
    complex_count = num_samples - positive - negative
    return {
        'positive_fraction': positive / num_samples,
        'negative_fraction': negative / num_samples,
        'complex_fraction': complex_count / num_samples,
        'average_sign': np.mean(np.sign(real_parts)),
        'sign_problem_severity': 1.0 - np.abs(np.mean(np.sign(real_parts)))
    }


if __name__ == "__main__":
    print("Tensor Network Utilities Test")
    mps = create_random_mps(num_sites=4, bond_dim=4, seed=42)
    print(f"Created MPS with {len(mps.tensors)} sites, bond dims: {mps.bond_dims}")
    ghz = create_ghz_mps(num_sites=4)
    entropy = compute_bond_entropy(ghz, bond=1)
    print(f"GHZ entanglement entropy: {entropy:.4f}")
