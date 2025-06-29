#!/usr/bin/env python3

"""
Quantum Zeno Effect Demonstration

This script demonstrates the Quantum Zeno Effect by repeatedly measuring a qubit
in a gradually rotated basis. Frequent measurements in a changing basis force
the quantum state to transition from ket(0) to ket(1).

Key parameters:
- THETA_DEG: Basis rotation angle in degrees (small angle like 2 degrees)
- NUM_MEASUREMENTS: Number of measurements needed for 90 degree total rotation
- NUM_ATTEMPTS: Number of experimental runs for statistics
"""

import argparse
import random

import matplotlib.pyplot as plt
import numpy as np
from qiskit.quantum_info import Statevector

from .utils import validate_filename

THETA_DEG = 2  # Basis rotation angle in degrees
THETA_RAD = np.radians(THETA_DEG)  # Converted to radians
NUM_ATTEMPTS = 1000  # Number of experimental runs
NUM_MEASUREMENTS = int(
    np.radians(90) / THETA_RAD
)  # Measurements needed for 90 degree rotation


def measurement_projectors(theta):
    """Create projection operators for measurement in rotated basis

    Returns:
        tuple: (P_v, P_w) where P_v = ket(v) * bra(v) and P_w = ket(w) * bra(w)
               for the rotated basis states v and w
    """
    v = np.array([np.cos(theta), np.sin(theta)])
    w = np.array([-np.sin(theta), np.cos(theta)])
    P_v = np.outer(v, v)  # ket(v) * bra(v)
    P_w = np.outer(w, w)  # ket(w) * bra(w)
    return P_v, P_w


def measure_single_qzeno(state, theta_rad, num_measurements):
    """Perform sequence of measurements in rotated basis

    Args:
        state (Statevector): Initial quantum state
        theta_rad (float): Rotation angle per measurement in radians
        num_measurements (int): Number of measurements to perform

    Returns:
        Statevector: Quantum state after measurements
    """
    P_v, P_w = measurement_projectors(theta_rad)

    for _ in range(num_measurements):
        # Calculate probabilities
        prob_v = np.abs(state.data.conj() @ P_v @ state.data)
        prob_w = 1 - prob_v

        # Collapse state based on measurement
        if np.random.random() < prob_v:
            state = Statevector((P_v @ state.data) / np.sqrt(prob_v))
        else:
            state = Statevector((P_w @ state.data) / np.sqrt(prob_w))

    return state


def qzeno(
    theta_rad=THETA_RAD,
    num_measurements=None,
    num_attempts=NUM_ATTEMPTS,
    random_seed=None,
):
    """Run quantum Zeno experiment"""
    if random_seed is not None:
        np.random.seed(random_seed)
        random.seed(random_seed)

    # Validate input parameters
    if theta_rad <= 0:
        raise ValueError("theta_rad must be positive")
    if num_attempts <= 0:
        raise ValueError("num_attempts must be positive")

    # Calculate number of measurements if not specified
    if num_measurements is None:
        if theta_rad < 1e-5:  # Avoid division by zero for very small angles
            raise ValueError(
                "theta_rad must be >= 1e-5 to have reasonable number of measurements"
            )
        num_measurements = int(np.radians(90) / theta_rad)

    results = [0, 0]

    for _ in range(num_attempts):
        state = Statevector([1, 0])
        state = measure_single_qzeno(state, theta_rad, num_measurements)

        prob_0 = np.abs(state.data[0]) ** 2
        results[int(np.random.random() >= prob_0)] += 1

    return results


def main(output_file):
    results = qzeno()
    print("Qzeno measurement counts:", results)

    # Create and save visualization
    plt.figure(figsize=(8, 5))
    plt.bar(["ket(0)", "ket(1)"], results, color=["blue", "red"])
    plt.title(
        f"Quantum Zeno Effect\n({NUM_MEASUREMENTS} measurements at {THETA_DEG}° increments)"
    )
    plt.ylabel("Measurement counts")
    plt.savefig(output_file)
    plt.close()
    print(f"Histogram saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demonstrate Quantum Zeno Effect.")
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="qzeno.png",
        help="Output image path (default: qzeno.png)",
    )
    args = parser.parse_args()

    output_file = validate_filename(args.output, "png")
    main(output_file)
