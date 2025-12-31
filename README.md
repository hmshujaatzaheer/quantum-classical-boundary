# Quantum-Classical Boundary Research Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

A comprehensive framework for characterizing quantum-classical computational boundaries using machine learning approaches to trainability, simulability, and verification.

**Author:** H M Shujaat Zaheer  
**Institution:** PhD Research - ETH Zurich Quantum Computing Group (Prof. Dominik Hangleiter)  
**Repository:** https://github.com/hmshujaatzaheer/quantum-classical-boundary

## Overview

This repository implements the algorithms and experiments from the PhD proposal "Characterizing Quantum-Classical Computational Boundaries: Machine Learning Approaches to Trainability, Simulability, and Verification."

### Three Research Thrusts

1. **QUANTUM-PHASE-GNN** (`src/quantum_phase_gnn/`): ML-based boundary classification using Graph Neural Networks
2. **POSITIVITY-GAUGE-OPT** (`src/positivity_gauge_opt/`): GNN-guided tensor network optimization for sign-problem-free simulation
3. **BELL-VERIFY-NEAR-TERM** (`src/bell_verify/`): Practical verification protocols for NISQ devices

## Installation

```bash
git clone https://github.com/hmshujaatzaheer/quantum-classical-boundary.git
cd quantum-classical-boundary
pip install -r requirements.txt
```

## Quick Start

### Algorithm 1: DLA Computation

```python
from src.quantum_phase_gnn import DLAComputer, extract_circuit_generators

# Define a quantum circuit
circuit = [
    {'gate': 'Ry', 'qubits': [0]},
    {'gate': 'Ry', 'qubits': [1]},
    {'gate': 'CNOT', 'qubits': [0, 1]},
]

# Compute DLA dimension
dla = DLAComputer(n_qubits=2)
generators = extract_circuit_generators(circuit, n_qubits=2)
dim = dla.compute_dimension(generators)
print(f"DLA dimension: {dim}")
```

### Algorithm 2: Phase Classification

```python
from src.quantum_phase_gnn import QuantumPhaseGNN

gnn = QuantumPhaseGNN()
result = gnn.classify(circuit, n_qubits=2)
print(f"Phase: {result.predicted_phase.name}")
print(f"Trainability: {result.trainability_score:.4f}")
```

### Algorithm 3: Gauge Optimization

```python
from src.positivity_gauge_opt import PositivityGaugeOptimizer, create_random_mps

mps = create_random_mps(num_sites=6, bond_dim=4)
optimizer = PositivityGaugeOptimizer()
result = optimizer.optimize(mps.tensors)
print(f"Positivity: {result.positivity_score:.4f}")
print(f"Phase: {result.phase.value}")
```

### Algorithm 4: Bell Verification

```python
from src.bell_verify import BellVerificationProtocol, DeviceCharacteristics
import numpy as np

device = DeviceCharacteristics(
    single_qubit_error=0.001, two_qubit_error=0.01,
    readout_error=0.02, t1_time=100e-6, t2_time=80e-6, gate_time=50e-9
)

protocol = BellVerificationProtocol()
# samples and ideal_probs would come from actual quantum device
result = protocol.verify(samples, ideal_probs, device)
print(f"Outcome: {result.outcome.value}")
```

## Repository Structure

```
quantum-classical-boundary/
├── src/
│   ├── quantum_phase_gnn/      # Research Thrust 1
│   │   ├── dla.py              # Algorithm 1: DLA-COMPUTE
│   │   ├── model.py            # Algorithm 2: QUANTUM-PHASE-GNN
│   │   ├── features.py         # Feature extraction
│   │   └── states.py           # Quantum state utilities
│   ├── positivity_gauge_opt/   # Research Thrust 2
│   │   ├── gauge_optimizer.py  # Algorithm 3: POSITIVITY-GAUGE-OPT
│   │   └── tensor_network.py   # Tensor network utilities
│   └── bell_verify/            # Research Thrust 3
│       ├── protocol.py         # Algorithm 4: BELL-VERIFY-NEAR-TERM
│       └── noise_estimation.py # Noise modeling
├── proofs/                     # Mathematical proofs
│   ├── approximation_guarantee/
│   ├── convergence_analysis/
│   └── sample_complexity/
├── experiments/                # Reproducible experiments
│   ├── meyer_verification/     # Meyer et al. validation
│   ├── dla_experiments/        # DLA scaling analysis
│   ├── positivity_experiments/ # Gauge optimization
│   └── bell_experiments/       # Sample complexity
├── docs/                       # Documentation
└── tests/                      # Unit tests
```

## Key References

- Meyer et al. (2025) "Exploiting structure in quantum computing"
- Ragone et al. (2024) "A Lie algebraic theory of barren plateaus"
- Hangleiter et al. (2024) "Computational advantage from Bell sampling"
- Ferracin et al. (2024) "Efficiently verifying quantum computation"

## License

MIT License - see LICENSE file for details.

## Citation

```bibtex
@misc{zaheer2025quantum,
  title={Characterizing Quantum-Classical Computational Boundaries},
  author={Zaheer, H M Shujaat},
  year={2025},
  institution={ETH Zurich}
}
```
