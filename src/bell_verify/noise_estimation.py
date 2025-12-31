"""
Noise Estimation Utilities
==========================

Utilities for characterizing and modeling NISQ device noise.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class NoiseParameters:
    """Complete noise model parameters."""
    depolarizing_rate: float
    amplitude_damping_rate: float
    dephasing_rate: float
    readout_error_0: float
    readout_error_1: float
    crosstalk_strength: float


def depolarizing_channel(rho: np.ndarray, p: float) -> np.ndarray:
    """Apply depolarizing channel: ρ → (1-p)ρ + p·I/d"""
    d = rho.shape[0]
    return (1 - p) * rho + p * np.eye(d) / d


def amplitude_damping_channel(rho: np.ndarray, gamma: float) -> np.ndarray:
    """Apply amplitude damping channel (T1 decay)."""
    K0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]])
    K1 = np.array([[0, np.sqrt(gamma)], [0, 0]])
    return K0 @ rho @ K0.conj().T + K1 @ rho @ K1.conj().T


def dephasing_channel(rho: np.ndarray, gamma: float) -> np.ndarray:
    """Apply dephasing channel (T2 decay)."""
    K0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]])
    K1 = np.array([[0, 0], [0, np.sqrt(gamma)]])
    return K0 @ rho @ K0.conj().T + K1 @ rho @ K1.conj().T


def readout_error_channel(probs: np.ndarray, p0: float, p1: float) -> np.ndarray:
    """Apply readout errors: p0 = P(1|0), p1 = P(0|1)."""
    confusion = np.array([[1 - p0, p1], [p0, 1 - p1]])
    if probs.ndim == 1:
        return confusion @ probs
    return confusion @ probs


def estimate_noise_from_rb(rb_data: List[Tuple[int, float]]) -> Dict:
    """Estimate noise parameters from randomized benchmarking data."""
    if len(rb_data) < 2:
        return {'depolarizing_rate': 0.01, 'error_per_gate': 0.001}
    depths = np.array([d[0] for d in rb_data])
    fidelities = np.array([d[1] for d in rb_data])
    log_fid = np.log(np.maximum(fidelities, 1e-10))
    slope, intercept = np.polyfit(depths, log_fid, 1)
    p = 1 - np.exp(slope)
    r = p * (1 - 1/2)
    return {
        'depolarizing_rate': p,
        'error_per_gate': r,
        'spam_error': 1 - np.exp(intercept),
        'fit_quality': np.corrcoef(depths, log_fid)[0, 1] ** 2
    }


def estimate_t1_t2(relaxation_data: List[Tuple[float, float]],
                   dephasing_data: List[Tuple[float, float]]) -> Dict:
    """Estimate T1 and T2 from decay experiments."""
    times_t1 = np.array([d[0] for d in relaxation_data])
    probs_t1 = np.array([d[1] for d in relaxation_data])
    log_probs = np.log(np.maximum(probs_t1, 1e-10))
    t1_inv, _ = np.polyfit(times_t1, log_probs, 1)
    t1 = -1 / t1_inv if t1_inv != 0 else float('inf')
    
    times_t2 = np.array([d[0] for d in dephasing_data])
    coherence = np.array([d[1] for d in dephasing_data])
    log_coh = np.log(np.maximum(coherence, 1e-10))
    t2_inv, _ = np.polyfit(times_t2, log_coh, 1)
    t2 = -1 / t2_inv if t2_inv != 0 else float('inf')
    
    return {'t1': t1, 't2': t2, 't2_star': t2 / 2, 'quality_factor': t2 / (2 * t1) if t1 > 0 else 0}


def calibrate_readout(calibration_shots: int = 1000) -> Tuple[float, float]:
    """Simulate readout calibration (placeholder for real device)."""
    p0 = 0.02 + 0.01 * np.random.randn()
    p1 = 0.03 + 0.01 * np.random.randn()
    return max(0, min(1, p0)), max(0, min(1, p1))


def build_noise_model(device_params: Dict) -> NoiseParameters:
    """Build complete noise model from device parameters."""
    t1 = device_params.get('t1', 100e-6)
    t2 = device_params.get('t2', 80e-6)
    gate_time = device_params.get('gate_time', 50e-9)
    gamma_1 = 1 - np.exp(-gate_time / t1)
    gamma_2 = 1 - np.exp(-gate_time / t2)
    
    return NoiseParameters(
        depolarizing_rate=device_params.get('depolarizing', 0.001),
        amplitude_damping_rate=gamma_1,
        dephasing_rate=gamma_2 - gamma_1 / 2,
        readout_error_0=device_params.get('readout_0', 0.02),
        readout_error_1=device_params.get('readout_1', 0.03),
        crosstalk_strength=device_params.get('crosstalk', 0.001)
    )


def simulate_noisy_measurement(ideal_state: np.ndarray, noise: NoiseParameters,
                                num_shots: int = 1000) -> np.ndarray:
    """Simulate noisy measurement outcomes."""
    rho = np.outer(ideal_state, np.conj(ideal_state))
    rho = depolarizing_channel(rho, noise.depolarizing_rate)
    ideal_probs = np.real(np.diag(rho))
    noisy_probs = readout_error_channel(ideal_probs, noise.readout_error_0, noise.readout_error_1)
    noisy_probs = np.maximum(noisy_probs, 0)
    noisy_probs /= np.sum(noisy_probs)
    return np.random.choice(len(noisy_probs), size=num_shots, p=noisy_probs)


def compute_effective_noise_rate(noise: NoiseParameters, circuit_depth: int,
                                  n_qubits: int) -> float:
    """Compute effective noise rate for a circuit."""
    single_q_noise = noise.depolarizing_rate * n_qubits * circuit_depth
    decoherence = (noise.amplitude_damping_rate + noise.dephasing_rate) * n_qubits * circuit_depth
    readout = (noise.readout_error_0 + noise.readout_error_1) / 2 * n_qubits
    return min(single_q_noise + decoherence + readout, 1.0)


if __name__ == "__main__":
    print("Noise Estimation Module Test")
    rb_data = [(1, 0.99), (10, 0.90), (50, 0.60), (100, 0.35)]
    noise_params = estimate_noise_from_rb(rb_data)
    print(f"Estimated depolarizing rate: {noise_params['depolarizing_rate']:.4f}")
    print(f"Error per gate: {noise_params['error_per_gate']:.4f}")
