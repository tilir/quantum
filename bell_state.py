#!/usr/bin/env python3

# This script creates and simulates a quantum circuit that demonstrates 
# the creation of an entangled Bell state using Hadamard and CNOT gates.
# The result of 1000 runs is measured and visualized as a histogram.

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
from qiskit.quantum_info import Statevector
from qiskit.quantum_info import Operator
from qiskit.visualization import array_to_latex
import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys

def validate_filename(filename):
    """Validate the output filename and ensure it has .png extension"""
    if not filename:
        print("Error: Output filename cannot be empty", file=sys.stderr)
        sys.exit(1)
    return filename if filename.lower().endswith('.png') else f"{filename}.png"

def print_statevector(qc, description):
    """Print current statevector with description"""
    state = Statevector(qc)
    print(f"\n{description}:")
    print("Statevector:", state.draw(output='text'))

def print_opmatrix(qc, description):
    op = Operator(qc)
    m = op.data
    print(f"\n{description}:")
    print("Opmatrix:\n", np.round(m, 4))

def main(output_file):
    # Create a circuit with 2 qubits and 2 classical bits
    qc = QuantumCircuit(2, 2)
    print_statevector(qc, "Initial state")

    # Step 1: Apply the Hadamard gate (H) to the first qubit to create superposition
    qc.h(0)  # H gate on qubit q0: ket(0) -> (ket(0) + ket(1)) / sqrt(2)
    print_opmatrix(qc, "Matrix after H on q0")
    print_statevector(qc, "After H on q0")

    # Step 2: Apply a CNOT gate with q0 as control and q1 as target
    qc.cx(0, 1)  # CNOT flips q1 if q0 is ket(1), creating an entangled state
    print_opmatrix(qc, "Matrix after CNOT")
    print_statevector(qc, "After CNOT")

    # Step 3: Measure both qubits and store the results in classical bits
    qc.measure([0, 1], [0, 1])  # Measure q0 -> c0, q1 -> c1

    # Step 4: Run the simulation
    simulator = AerSimulator()
    job = simulator.run(qc, shots=1000)  # Perform 1000 runs
    result = job.result()

    # Step 5: Retrieve measurement results
    counts = result.get_counts(qc)
    print("Measurement results:", counts)

    # Step 6: Visualize and save the results
    fig = plot_histogram(counts)
    fig.savefig(output_file)
    print(f"Histogram saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate and visualize Bell state measurements.')
    parser.add_argument('--output', '-o', type=str, default='bell_state.png',
                       help='Output PNG file path (default: bell_state.png)')
    args = parser.parse_args()
    
    # Validate and normalize the filename
    output_file = validate_filename(args.output)
    main(output_file)
