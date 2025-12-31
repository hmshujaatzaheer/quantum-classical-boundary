"""
BELL-VERIFY-NEAR-TERM Package
=============================

Verification protocols for near-term quantum devices.
Implements Research Thrust 3: Practical Verification.
"""

from .protocol import (
    BellVerificationProtocol,
    VerificationOutcome,
    VerificationResult,
    DeviceCharacteristics,
    PartialFidelityEstimator,
    adaptive_verification
)
from .noise_estimation import (
    NoiseParameters,
    depolarizing_channel,
    amplitude_damping_channel,
    dephasing_channel,
    readout_error_channel,
    estimate_noise_from_rb,
    estimate_t1_t2,
    build_noise_model,
    simulate_noisy_measurement,
    compute_effective_noise_rate
)

__all__ = [
    'BellVerificationProtocol', 'VerificationOutcome', 'VerificationResult',
    'DeviceCharacteristics', 'PartialFidelityEstimator', 'adaptive_verification',
    'NoiseParameters', 'depolarizing_channel', 'amplitude_damping_channel',
    'dephasing_channel', 'readout_error_channel', 'estimate_noise_from_rb',
    'estimate_t1_t2', 'build_noise_model', 'simulate_noisy_measurement',
    'compute_effective_noise_rate'
]

__version__ = '0.1.0'
