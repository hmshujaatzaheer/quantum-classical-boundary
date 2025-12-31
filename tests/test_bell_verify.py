"""Tests for Bell verification module."""

import numpy as np
import sys
sys.path.insert(0, '..')
from src.bell_verify import (
    BellVerificationProtocol,
    VerificationOutcome,
    DeviceCharacteristics,
    adaptive_verification
)


def test_verification_outcomes():
    """Test that verification returns valid outcomes."""
    np.random.seed(42)
    n_qubits = 4
    dim = 2 ** n_qubits
    
    ideal_probs = np.random.dirichlet(np.ones(dim))
    samples = np.random.choice(dim, size=1000, p=ideal_probs)
    
    device = DeviceCharacteristics(
        single_qubit_error=0.001, two_qubit_error=0.01,
        readout_error=0.02, t1_time=100e-6, t2_time=80e-6, gate_time=50e-9
    )
    
    protocol = BellVerificationProtocol()
    result = protocol.verify(samples, ideal_probs, device)
    
    assert result.outcome in list(VerificationOutcome)
    assert 0 <= result.estimated_fidelity <= 1
    assert result.num_samples == 1000


def test_confidence_interval():
    """Test confidence interval is valid."""
    np.random.seed(42)
    n_qubits = 4
    dim = 2 ** n_qubits
    ideal_probs = np.random.dirichlet(np.ones(dim))
    samples = np.random.choice(dim, size=1000, p=ideal_probs)
    
    device = DeviceCharacteristics(
        single_qubit_error=0.001, two_qubit_error=0.01,
        readout_error=0.02, t1_time=100e-6, t2_time=80e-6, gate_time=50e-9
    )
    
    protocol = BellVerificationProtocol()
    result = protocol.verify(samples, ideal_probs, device)
    
    lower, upper = result.confidence_interval
    assert lower <= upper
    assert 0 <= lower <= 1
    assert 0 <= upper <= 1


def test_adaptive_verification():
    """Test adaptive parameter computation."""
    device = DeviceCharacteristics(
        single_qubit_error=0.001, two_qubit_error=0.01,
        readout_error=0.02, t1_time=100e-6, t2_time=80e-6, gate_time=50e-9
    )
    
    params = adaptive_verification(circuit_depth=50, n_qubits=8, device=device)
    
    assert params['recommended_samples'] > 0
    assert 0 <= params['expected_fidelity'] <= 1
    assert isinstance(params['verification_feasible'], bool)


if __name__ == "__main__":
    test_verification_outcomes()
    test_confidence_interval()
    test_adaptive_verification()
    print("All Bell verification tests passed!")
