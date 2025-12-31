# Getting Started Guide

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Basic Installation

```bash
git clone https://github.com/hmshujaatzaheer/quantum-classical-boundary.git
cd quantum-classical-boundary
pip install -r requirements.txt
```

### Development Installation

```bash
pip install -e .
```

## Quick Tutorial

### 1. Computing DLA Dimension

The Dynamical Lie Algebra (DLA) dimension is a key indicator of circuit trainability:

```python
from src.quantum_phase_gnn import DLAComputer, extract_circuit_generators

# Create a simple circuit
circuit = [
    {'gate': 'Ry', 'qubits': [0]},
    {'gate': 'Ry', 'qubits': [1]},
    {'gate': 'CNOT', 'qubits': [0, 1]},
]

dla = DLAComputer(n_qubits=2)
generators = extract_circuit_generators(circuit, n_qubits=2)
dim = dla.compute_dimension(generators)

print(f"DLA dimension: {dim}")
print(f"Max possible: {4**2 - 1}")
print(f"DLA ratio: {dim / (4**2 - 1):.4f}")
```

### 2. Phase Classification

Classify circuits into trainable/transition/barren plateau phases:

```python
from src.quantum_phase_gnn import QuantumPhaseGNN

gnn = QuantumPhaseGNN()
result = gnn.classify(circuit, n_qubits=2)

print(f"Predicted phase: {result.predicted_phase.name}")
print(f"Confidence: {result.confidence:.2%}")
print(f"DLA ratio: {result.dla_ratio:.4f}")
```

### 3. Tensor Network Optimization

Optimize gauge transformations for sign-free simulation:

```python
from src.positivity_gauge_opt import (
    PositivityGaugeOptimizer, 
    create_random_mps
)

# Create random MPS
mps = create_random_mps(num_sites=6, bond_dim=4, seed=42)

# Optimize
optimizer = PositivityGaugeOptimizer(max_iterations=200)
result = optimizer.optimize(mps.tensors)

print(f"Final positivity: {result.positivity_score:.4f}")
print(f"Phase: {result.phase.value}")
print(f"Converged: {result.converged}")
```

### 4. Verification Protocol

Verify quantum computation results:

```python
from src.bell_verify import (
    BellVerificationProtocol,
    DeviceCharacteristics,
    adaptive_verification
)
import numpy as np

# Define device characteristics
device = DeviceCharacteristics(
    single_qubit_error=0.001,
    two_qubit_error=0.01,
    readout_error=0.02,
    t1_time=100e-6,
    t2_time=80e-6,
    gate_time=50e-9
)

# Get recommended parameters
params = adaptive_verification(
    circuit_depth=50, 
    n_qubits=8, 
    device=device
)
print(f"Recommended samples: {params['recommended_samples']}")
print(f"Expected fidelity: {params['expected_fidelity']:.4f}")
```

## Running Experiments

### Meyer Verification

```bash
cd experiments/meyer_verification
python verify_meyer_findings.py
```

### DLA Scaling

```bash
cd experiments/dla_experiments
python dla_scaling.py
```

### Gauge Convergence

```bash
cd experiments/positivity_experiments
python gauge_convergence.py
```

### Sample Complexity

```bash
cd experiments/bell_experiments
python sample_complexity.py
```

## Next Steps

- Read the mathematical proofs in `proofs/`
- Explore the API documentation
- Run the unit tests: `pytest tests/`
