#!/usr/bin/env python3
"""Demonstration of entangled Bell state creation using Hadamard and CNOT gates.

This script creates a quantum circuit that generates a Bell state by applying
Hadamard and CNOT gates, then simulates and visualizes the measurement results.
The output shows the characteristic 50/50 distribution between ket(00) and ket(11) states.
"""

import argparse

from qiskit.visualization import circuit_drawer, plot_histogram
from qiskit_aer import AerSimulator

from .qstates import bell_state
from .utils import validate_filename


def simulate_circuit(circuit_file=None):
    qc = bell_state()

    # Measure qubits
    qc.measure([0, 1], [0, 1])

    if circuit_file:
        circuit_drawer(qc, output="mpl", filename=circuit_file)
        print(f"Bell: circuit rendered to {circuit_file}")

    # Execute simulation
    simulator = AerSimulator()
    result = simulator.run(qc, shots=1000).result()
    return result.get_counts(qc)


def main(output_file, circuit_file):
    """Executes Bell state experiment and saves results.

    Args:
        output_file (str): Path to save the measurement histogram.
        output_file (str): Path to save the quantum circuit picture.
    """
    counts = simulate_circuit(circuit_file)

    print("Bell: measurement results:", counts)

    plot_histogram(counts).savefig(output_file)
    print(f"Bell: results rendered to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate and visualize Bell state measurements."
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="bell_state.png",
        help="Output diag file path (default: bell_state.png)",
    )
    parser.add_argument(
        "--circuit",
        "-c",
        type=str,
        default="bell_circuit.svg",
        help="Output circuit file path (default: bell_circuit.svg)",
    )
    args = parser.parse_args()
    outf = validate_filename(args.output, "png")
    cf = validate_filename(args.circuit, "svg")
    main(outf, cf)
