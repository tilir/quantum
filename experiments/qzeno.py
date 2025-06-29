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


def measurement_projectors(v, w):
    """Create projection operators for given vectors (1-qubit case)

    Returns:
        tuple: (P_v, P_w) where P_v = ket(v) * bra(v) and P_w = ket(w) * bra(w)
               for the rotated basis states v and w
    """
    P_v = np.outer(v, v.conj())  # ket(v) * bra(v)
    P_w = np.outer(w, w.conj())  # ket(w) * bra(w)
    return P_v, P_w


def probs_normalized(state, projectors):
    # Calculate probabilities
    probs = [np.abs(state.conj() @ P @ state) for P in projectors]
    
    # Normalize probabilities (might not sum to 1 due to precision)
    total_prob = sum(probs)
    probs = [p / total_prob for p in probs]
    return probs


def measure_in_basis(state, v, w):
    """Measures 1-qubit state in <v, w> basis

    Returns:
        collapsed state
    """
    P_v, P_w = measurement_projectors(v, w)
    prob_v, prob_w = probs_normalized(state, [P_v, P_w])

    rand_val = np.random.random()
    if rand_val < prob_v:
        state = (P_v @ state) / np.sqrt(prob_v)
        print(f"P_v: rand={rand_val:.2f}, prob_v={prob_v:.2f}")
    else:
        state = (P_w @ state) / np.sqrt(prob_w)
        print(f"P_w: rand={rand_val:.2f}, prob_w={prob_w:.2f}")
    return state


def qzeno_measurements(state, dtheta, num_measurements):
    """Perform sequence of measurements in rotated basis

    Args:
        state (Statevector): Initial quantum state
        dtheta (float): Rotation angle per measurement in radians
        num_measurements (int): Number of measurements to perform

    Returns:
        Statevector: Quantum state after measurements
    """
    theta = dtheta

    for _ in range(num_measurements):
      v = np.array([np.cos(theta), np.sin(theta)])
      w = np.array([-np.sin(theta), np.cos(theta)])
      state = measure_in_basis(state, v, w)
      theta += dtheta

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
    z = np.array([1, 0])
    o = np.array([0, 1])
    P_0, P_1 = measurement_projectors(z, o)

    for _ in range(num_attempts):
        state = np.array([1, 0])
        state = qzeno_measurements(state, theta_rad, num_measurements)
        probs = probs_normalized(state, [P_0, P_1])
        prob_0 = probs[0]

        #print("state: ", state)
        #print("p0: ", prob_0)
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
