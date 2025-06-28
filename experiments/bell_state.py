#!/usr/bin/env python3
"""Demonstration of entangled Bell state creation using Hadamard and CNOT gates.

This script creates a quantum circuit that generates a Bell state by applying
Hadamard and CNOT gates, then simulates and visualizes the measurement results.
The output shows the characteristic 50/50 distribution between ket(00) and ket(11) states.
"""

import argparse

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, Statevector
from qiskit.visualization import plot_histogram
from qiskit_aer import AerSimulator

from .utils import validate_filename

VERBOSE = 0  # Controls debug output verbosity (0/1)


def print_statevector(qc, description):
    """Prints the current quantum statevector with contextual description.

    Args:
        qc (QuantumCircuit): Quantum circuit to analyze.
        description (str): Contextual information about the state being printed.

    Returns:
        None: Outputs directly to stdout.

    Example:
        >>> circuit = QuantumCircuit(2)
        >>> print_statevector(circuit, "Initial state")
        Initial state:
        Statevector: [1.+0.j 0.+0.j 0.+0.j 0.+0.j]
    """
    state = Statevector(qc)
    print(f"\n{description}:")
    print("Statevector:", state.draw(output="text"))


def print_opmatrix(qc, description):
    """Prints the unitary operator matrix for the current circuit state.

    Args:
        qc (QuantumCircuit): Quantum circuit to analyze.
        description (str): Context for when this matrix is being printed.
    """
    op = Operator(qc)
    print(f"\n{description}:")
    print("Opmatrix:\n", np.round(op.data, 4))


def bell_state():
    """Creates and measures a Bell state quantum circuit.

    Returns:
        dict: Measurement counts dictionary with keys '00' and '11'.

    The circuit implements:
    1. Hadamard gate on qubit 0 to create superposition
    2. CNOT gate between qubits 0 (control) and 1 (target)
    3. Measurement of both qubits
    """
    qc = QuantumCircuit(2, 2)
    print_statevector(qc, "Initial state")

    # Create superposition
    qc.h(0)
    if VERBOSE:
        print_opmatrix(qc, "Matrix after H on q0")
        print_statevector(qc, "After H on q0")

    # Create entanglement
    qc.cx(0, 1)
    if VERBOSE:
        print_opmatrix(qc, "Matrix after CNOT")
        print_statevector(qc, "After CNOT")

    # Measure qubits
    qc.measure([0, 1], [0, 1])

    # Execute simulation
    simulator = AerSimulator()
    result = simulator.run(qc, shots=1000).result()
    return result.get_counts(qc)


def main(output_file):
    """Executes Bell state experiment and saves results.

    Args:
        output_file (str): Path to save the measurement histogram.
    """
    counts = bell_state()
    print("Measurement results:", counts)

    plot_histogram(counts).savefig(output_file)
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate and visualize Bell state measurements."
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="bell_state.png",
        help="Output PNG file path (default: bell_state.png)",
    )
    args = parser.parse_args()
    main(validate_filename(args.output))
