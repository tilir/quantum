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

from .utils import validate_filename
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
from qiskit.quantum_info import Statevector
from qiskit.quantum_info import Operator
from qiskit.visualization import array_to_latex
import matplotlib.pyplot as plt
import numpy as np
import argparse

THETA_DEG = 2  # Basis rotation angle in degrees
THETA_RAD = np.radians(THETA_DEG)  # Converted to radians
NUM_ATTEMPTS = 1000  # Number of experimental runs
NUM_MEASUREMENTS = int(np.radians(90) / THETA_RAD)  # Measurements needed for 90 degree rotation

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

def main(output_file):
    """Run quantum Zeno experiment and save results to file
    
    Args:
        output_file (str): Path to save output visualization
    """
    P_v, P_w = measurement_projectors(THETA_RAD)
    results = [0, 0]  # [count_0, count_1]

    for _ in range(NUM_ATTEMPTS):
        # Initialize to ket(0)
        state = Statevector([1, 0])
        
        # Perform sequence of measurements in rotated basis
        for _ in range(NUM_MEASUREMENTS):
            # Calculate measurement probabilities
            prob_v = np.abs(state.data.conj() @ P_v @ state.data)
            prob_w = 1 - prob_v  # Because P_v + P_w = identity
            
            # Collapse state based on measurement outcome
            if np.random.random() < prob_v:
                state = Statevector((P_v @ state.data)/np.sqrt(prob_v))
            else:
                state = Statevector((P_w @ state.data)/np.sqrt(prob_w))

        # Final measurement in standard basis
        prob_0 = np.abs(state.data[0])**2
        results[np.random.random() >= prob_0] += 1

    print("Final measurement counts:", results)

    # Create and save visualization
    plt.figure(figsize=(8, 5))
    plt.bar(['ket(0)', 'ket(1)'], results, color=['blue', 'red'])
    plt.title(f"Quantum Zeno Effect\n({NUM_MEASUREMENTS} measurements at {THETA_DEG}° increments)")
    plt.ylabel("Measurement counts")
    plt.savefig(output_file)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Demonstrate Quantum Zeno Effect.')
    parser.add_argument('--output', '-o', type=str, default='qzeno.png',
                       help='Output image path (default: qzeno.png)')
    args = parser.parse_args()
    
    output_file = validate_filename(args.output)
    main(output_file)