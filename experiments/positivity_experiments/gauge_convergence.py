"""
Gauge Optimization Convergence Experiments
==========================================

Experiments analyzing convergence of POSITIVITY-GAUGE-OPT algorithm.
"""

import numpy as np
import time
import sys
sys.path.insert(0, '../..')
from src.positivity_gauge_opt import (
    PositivityGaugeOptimizer, TensorNetwork, create_random_mps,
    create_ghz_mps, analyze_positivity
)


def run_convergence_experiment():
    """Analyze convergence of gauge optimization."""
    print("=" * 60)
    print("Gauge Optimization Convergence Experiments")
    print("=" * 60)
    
    results = []
    
    for num_sites in [4, 6, 8]:
        for bond_dim in [2, 4, 8]:
            print(f"\n--- Sites={num_sites}, Bond dim={bond_dim} ---")
            
            mps = create_random_mps(num_sites, bond_dim, seed=42)
            initial_pos = analyze_positivity(mps)['positive_fraction']
            
            optimizer = PositivityGaugeOptimizer(
                learning_rate=0.1, max_iterations=200, tolerance=1e-6
            )
            
            start = time.time()
            result = optimizer.optimize(mps.tensors)
            elapsed = time.time() - start
            
            print(f"Initial positivity: {initial_pos:.4f}")
            print(f"Final positivity: {result.positivity_score:.4f}")
            print(f"Iterations: {result.iterations}")
            print(f"Converged: {result.converged}")
            print(f"Time: {elapsed:.3f}s")
            print(f"Phase: {result.phase.value}")
            
            results.append({
                'sites': num_sites, 'bond_dim': bond_dim,
                'initial': initial_pos, 'final': result.positivity_score,
                'iterations': result.iterations, 'time': elapsed
            })
    
    return results


def run_phase_transition_experiment():
    """Analyze positivity phase transitions."""
    print("\n" + "=" * 60)
    print("Positivity Phase Transition Analysis")
    print("=" * 60)
    
    num_sites = 6
    optimizer = PositivityGaugeOptimizer(max_iterations=100)
    
    for complexity in ['low', 'medium', 'high']:
        print(f"\n--- Complexity: {complexity} ---")
        
        if complexity == 'low':
            mps = create_ghz_mps(num_sites)
        elif complexity == 'medium':
            mps = create_random_mps(num_sites, bond_dim=4, seed=42)
        else:
            mps = create_random_mps(num_sites, bond_dim=8, seed=42)
            for i, t in enumerate(mps.tensors):
                mps.tensors[i] = t + 0.5j * np.random.randn(*t.shape)
        
        result = optimizer.optimize(mps.tensors)
        print(f"Final positivity: {result.positivity_score:.4f}")
        print(f"Phase: {result.phase.value}")


def run_scaling_benchmark():
    """Benchmark scaling with system size."""
    print("\n" + "=" * 60)
    print("Scaling Benchmark")
    print("=" * 60)
    
    bond_dim = 4
    optimizer = PositivityGaugeOptimizer(max_iterations=50)
    
    for num_sites in [4, 6, 8, 10, 12]:
        mps = create_random_mps(num_sites, bond_dim, seed=42)
        
        start = time.time()
        result = optimizer.optimize(mps.tensors)
        elapsed = time.time() - start
        
        print(f"Sites={num_sites}: Time={elapsed:.3f}s, Iterations={result.iterations}")


if __name__ == "__main__":
    run_convergence_experiment()
    run_phase_transition_experiment()
    run_scaling_benchmark()
