"""
Sample Complexity Experiments for BELL-VERIFY-NEAR-TERM
=======================================================

Experiments analyzing sample complexity and verification accuracy.
"""

import numpy as np
import sys
sys.path.insert(0, '../..')
from src.bell_verify import (
    BellVerificationProtocol, DeviceCharacteristics,
    adaptive_verification, simulate_noisy_measurement, build_noise_model
)


def run_sample_complexity_experiment():
    """Analyze sample complexity for different accuracy levels."""
    print("=" * 60)
    print("Sample Complexity Experiments")
    print("=" * 60)
    
    device = DeviceCharacteristics(
        single_qubit_error=0.001, two_qubit_error=0.01,
        readout_error=0.02, t1_time=100e-6, t2_time=80e-6, gate_time=50e-9
    )
    
    protocol = BellVerificationProtocol(confidence_level=0.95)
    
    results = []
    for n_qubits in [4, 6, 8, 10]:
        print(f"\n--- {n_qubits} qubits ---")
        dim = 2 ** n_qubits
        
        ideal_probs = np.random.dirichlet(np.ones(dim))
        ideal_probs /= np.sum(ideal_probs)
        
        for num_samples in [100, 500, 1000, 5000, 10000]:
            samples = np.random.choice(dim, size=num_samples, p=ideal_probs)
            result = protocol.verify(samples, ideal_probs, device)
            
            ci_width = result.confidence_interval[1] - result.confidence_interval[0]
            results.append({
                'n_qubits': n_qubits, 'samples': num_samples,
                'fidelity': result.estimated_fidelity, 'ci_width': ci_width,
                'outcome': result.outcome.value
            })
            
            print(f"N={num_samples}: F={result.estimated_fidelity:.4f}, "
                  f"CI width={ci_width:.4f}, Outcome={result.outcome.value}")
    
    return results


def run_noise_sensitivity_experiment():
    """Analyze sensitivity to device noise levels."""
    print("\n" + "=" * 60)
    print("Noise Sensitivity Analysis")
    print("=" * 60)
    
    n_qubits = 6
    dim = 2 ** n_qubits
    num_samples = 5000
    
    ideal_probs = np.random.dirichlet(np.ones(dim))
    protocol = BellVerificationProtocol()
    
    for error_rate in [0.001, 0.005, 0.01, 0.02, 0.05]:
        print(f"\n--- Two-qubit error rate: {error_rate} ---")
        
        device = DeviceCharacteristics(
            single_qubit_error=error_rate / 10,
            two_qubit_error=error_rate,
            readout_error=error_rate * 2,
            t1_time=100e-6, t2_time=80e-6, gate_time=50e-9
        )
        
        samples = np.random.choice(dim, size=num_samples, p=ideal_probs)
        result = protocol.verify(samples, ideal_probs, device)
        
        print(f"Fidelity: {result.estimated_fidelity:.4f}")
        print(f"XEB Score: {result.xeb_score:.4f}")
        print(f"Threshold: {result.threshold_used:.4f}")
        print(f"Outcome: {result.outcome.value}")


def run_adaptive_protocol_experiment():
    """Test adaptive verification protocol."""
    print("\n" + "=" * 60)
    print("Adaptive Protocol Analysis")
    print("=" * 60)
    
    device = DeviceCharacteristics(
        single_qubit_error=0.001, two_qubit_error=0.01,
        readout_error=0.02, t1_time=100e-6, t2_time=80e-6, gate_time=50e-9
    )
    
    for n_qubits in [4, 8, 12, 16]:
        for depth in [10, 50, 100]:
            params = adaptive_verification(depth, n_qubits, device)
            print(f"\nn={n_qubits}, depth={depth}:")
            print(f"  Recommended samples: {params['recommended_samples']}")
            print(f"  Expected fidelity: {params['expected_fidelity']:.4f}")
            print(f"  Feasible: {params['verification_feasible']}")


if __name__ == "__main__":
    np.random.seed(42)
    run_sample_complexity_experiment()
    run_noise_sensitivity_experiment()
    run_adaptive_protocol_experiment()
