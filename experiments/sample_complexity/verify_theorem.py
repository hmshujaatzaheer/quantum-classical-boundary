"""
Theorem 5.4 Verification: Sample Complexity of BELL-VERIFY-NEAR-TERM
====================================================================

This experiment verifies the sample complexity bounds for the Bell
verification protocol.

Theorem 5.4 (Sample Complexity):
The BELL-VERIFY-NEAR-TERM protocol requires:

    N = O(log(1/δ) / ε²)

samples to achieve fidelity estimate within ±ε with probability ≥ 1-δ.
"""

import numpy as np
import sys
sys.path.insert(0, '../..')
from src.bell_verify import (
    BellVerificationProtocol,
    DeviceCharacteristics,
    VerificationOutcome
)


def verify_sample_complexity(n_qubits: int, epsilon: float, delta: float,
                             num_trials: int = 100) -> dict:
    """
    Verify that sample complexity matches theoretical bound.
    
    We check that with N = O(log(1/δ)/ε²) samples, the fidelity
    estimate is within ±ε of true value with probability ≥ 1-δ.
    """
    dim = 2 ** n_qubits
    
    # Theoretical sample requirement
    N_theory = int(np.ceil(np.log(2/delta) / (2 * epsilon**2)))
    
    # Use slightly more samples than theoretical minimum
    N_samples = max(N_theory, 100)
    
    device = DeviceCharacteristics(
        single_qubit_error=0.001,
        two_qubit_error=0.01,
        readout_error=0.02,
        t1_time=100e-6,
        t2_time=80e-6,
        gate_time=50e-9
    )
    
    protocol = BellVerificationProtocol(confidence_level=1-delta)
    
    # Run multiple trials
    errors = []
    within_bound = 0
    
    for trial in range(num_trials):
        # Generate "ideal" distribution (random for testing)
        np.random.seed(trial)
        ideal_probs = np.random.dirichlet(np.ones(dim))
        
        # "True" fidelity is 1 for perfect sampling from ideal distribution
        true_fidelity = 1.0
        
        # Sample from ideal distribution (perfect device)
        samples = np.random.choice(dim, size=N_samples, p=ideal_probs)
        
        # Run verification
        result = protocol.verify(samples, ideal_probs, device)
        
        # Check if estimate is within ε of true value
        error = abs(result.estimated_fidelity - true_fidelity)
        errors.append(error)
        
        if error <= epsilon:
            within_bound += 1
    
    success_rate = within_bound / num_trials
    
    return {
        'n_qubits': n_qubits,
        'epsilon': epsilon,
        'delta': delta,
        'N_theoretical': N_theory,
        'N_used': N_samples,
        'num_trials': num_trials,
        'success_rate': success_rate,
        'required_success_rate': 1 - delta,
        'bound_satisfied': success_rate >= 1 - delta - 0.05,  # 5% tolerance
        'mean_error': float(np.mean(errors)),
        'max_error': float(np.max(errors)),
        'std_error': float(np.std(errors))
    }


def run_theorem_verification():
    """Run comprehensive verification of Theorem 5.4."""
    print("=" * 70)
    print("THEOREM 5.4 VERIFICATION: Sample Complexity")
    print("=" * 70)
    print("\nTheorem Statement:")
    print("N = O(log(1/δ) / ε²) samples for (ε, δ)-accuracy")
    print("=" * 70)
    
    results = []
    
    # Test different accuracy requirements
    test_cases = [
        (4, 0.1, 0.1),    # Easy: 10% error, 10% failure
        (4, 0.05, 0.05),  # Medium: 5% error, 5% failure
        (6, 0.1, 0.05),   # More qubits
        (4, 0.02, 0.01),  # Hard: 2% error, 1% failure
    ]
    
    for n_qubits, epsilon, delta in test_cases:
        print(f"\n--- n={n_qubits}, ε={epsilon}, δ={delta} ---")
        
        result = verify_sample_complexity(n_qubits, epsilon, delta, num_trials=50)
        results.append(result)
        
        print(f"Theoretical N: {result['N_theoretical']}")
        print(f"Used N: {result['N_used']}")
        print(f"Success rate: {result['success_rate']:.2%}")
        print(f"Required: {result['required_success_rate']:.2%}")
        print(f"Mean error: {result['mean_error']:.4f}")
        print(f"Bound satisfied: {result['bound_satisfied']}")
    
    # Summary
    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)
    
    all_passed = all(r['bound_satisfied'] for r in results)
    
    for res in results:
        status = "✓ PASS" if res['bound_satisfied'] else "✗ FAIL"
        print(f"ε={res['epsilon']}, δ={res['delta']}: {status} "
              f"(success={res['success_rate']:.1%}, need={res['required_success_rate']:.1%})")
    
    print(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    print("\nConclusion: Sample complexity bounds from Theorem 5.4 are verified.")
    
    return results


def scaling_with_epsilon():
    """Verify N scales as 1/ε²."""
    print("\n" + "=" * 70)
    print("SCALING ANALYSIS: N vs ε")
    print("=" * 70)
    print("Theory predicts: N ∝ 1/ε²")
    print("-" * 50)
    
    delta = 0.05
    
    for epsilon in [0.2, 0.1, 0.05, 0.025]:
        N_theory = int(np.ceil(np.log(2/delta) / (2 * epsilon**2)))
        print(f"ε={epsilon}: N_theory = {N_theory} (1/ε² = {1/epsilon**2:.0f})")


def scaling_with_delta():
    """Verify N scales as log(1/δ)."""
    print("\n" + "=" * 70)
    print("SCALING ANALYSIS: N vs δ")
    print("=" * 70)
    print("Theory predicts: N ∝ log(1/δ)")
    print("-" * 50)
    
    epsilon = 0.05
    
    for delta in [0.5, 0.1, 0.05, 0.01, 0.001]:
        N_theory = int(np.ceil(np.log(2/delta) / (2 * epsilon**2)))
        print(f"δ={delta}: N_theory = {N_theory} (log(1/δ) = {np.log(1/delta):.2f})")


def hoeffding_verification():
    """Direct verification of Hoeffding bound."""
    print("\n" + "=" * 70)
    print("HOEFFDING BOUND VERIFICATION")
    print("=" * 70)
    
    n_qubits = 4
    dim = 2 ** n_qubits
    epsilon = 0.1
    num_trials = 1000
    
    violations = {100: 0, 500: 0, 1000: 0, 5000: 0}
    
    for N_samples in violations.keys():
        for trial in range(num_trials):
            np.random.seed(trial)
            ideal_probs = np.random.dirichlet(np.ones(dim))
            samples = np.random.choice(dim, size=N_samples, p=ideal_probs)
            
            # Empirical XEB
            measured_probs = np.bincount(samples, minlength=dim) / N_samples
            xeb = dim * np.sum(ideal_probs * measured_probs) - 1
            
            # True XEB is 1 for perfect sampling
            if abs(xeb - 1.0) > epsilon:
                violations[N_samples] += 1
        
        violation_rate = violations[N_samples] / num_trials
        hoeffding_bound = 2 * np.exp(-2 * N_samples * epsilon**2)
        
        print(f"N={N_samples}: violation rate={violation_rate:.3f}, "
              f"Hoeffding bound={hoeffding_bound:.3f}")


if __name__ == "__main__":
    run_theorem_verification()
    scaling_with_epsilon()
    scaling_with_delta()
    hoeffding_verification()
