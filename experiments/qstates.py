from math import sqrt

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, Statevector

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
    print(f"{description}:\n")
    print(state.draw(output="text"))
    print("\n")


def print_opmatrix(qc, description):
    """Prints the unitary operator matrix for the current circuit state.

    Args:
        qc (QuantumCircuit): Quantum circuit to analyze.
        description (str): Context for when this matrix is being printed.
    """
    op = Operator(qc)
    print(f"{description}\n:")
    print(np.round(op.data, 4))
    print("\n")


def bell_state():
    """Creates a Bell state quantum circuit.

    Returns:
        QuantumCircuit: Configured circuit producing the target state

    The circuit implements:
    1. Hadamard gate on qubit 0 to create superposition
    2. CNOT gate between qubits 0 (control) and 1 (target)
    3. Measurement of both qubits
    """
    qc = QuantumCircuit(2, 2)
    if VERBOSE:
        print_statevector(qc, "Bell: Initial state")

    # Create superposition
    qc.h(0)
    if VERBOSE:
        print_opmatrix(qc, "Bell: Matrix after H on q0")
        print_statevector(qc, "Bell: After H on q0")

    # Create entanglement
    qc.cx(0, 1)
    if VERBOSE:
        print_opmatrix(qc, "Bell: Matrix after CNOT")
        print_statevector(qc, "Bell: After CNOT")

    return qc


def create_arbitrary_state(a, b, c, d):
    """Creates a quantum circuit preparing an arbitrary 2-qubit state.

    Constructs a circuit that produces the state: a*ket(00) + b*ket(01) + c*ket(10) + d*ket(11)
    with precise amplitude and phase control for all components.

    Args:
        a (complex): Amplitude for ket(00) state
        b (complex): Amplitude for ket(01) state
        c (complex): Amplitude for ket(10) state
        d (complex): Amplitude for ket(11) state

    Returns:
        QuantumCircuit: Configured circuit producing the target state

    Example:
        >>> # Create (ket(00) + 2j*ket(01) + 3*ket(10) + 4j*ket(11))/sqrt(30)
        >>> qc = create_arbitrary_state(1, 2j, 3, 4j)
    """
    # Normalize coefficients to ensure valid quantum state
    norm = sqrt(abs(a) ** 2 + abs(b) ** 2 + abs(c) ** 2 + abs(d) ** 2)
    a, b, c, d = a / norm, b / norm, c / norm, d / norm

    qc = QuantumCircuit(2)

    # Step 1: Prepare first qubit in superposition
    # theta1 controls ket(0) vs ket(1) probability distribution
    theta1 = 2 * np.arccos(np.sqrt(abs(a) ** 2 + abs(b) ** 2))
    # phi1 sets relative phase between ket(0) and ket(1) components
    phi1 = np.angle(b) - np.angle(a)
    qc.ry(theta1, 0)  # Rotation around Y-axis
    qc.rz(phi1, 0)  # Phase adjustment

    # Step 2: Conditional operations for second qubit
    # Normalized amplitudes for each branch
    alpha = a / sqrt(abs(a) ** 2 + abs(b) ** 2)  # ket(0) branch
    beta = c / sqrt(abs(c) ** 2 + abs(d) ** 2)  # ket(1) branch

    # Calculate rotation angles for each conditional branch
    theta2_0 = 2 * np.arctan2(abs(b), abs(a))  # If first qubit is ket(0)
    theta2_1 = 2 * np.arctan2(abs(d), abs(c))  # If first qubit is ket(1)

    # Apply controlled-Y rotations with X gates for conditional logic
    qc.cry(theta2_0, 0, 1)  # Control on q0, target q1 when q0=ket(0)
    qc.x(0)  # Flip to prepare ket(1) branch
    qc.cry(theta2_1 - theta2_0, 0, 1)  # Adjust for ket(1) branch
    qc.x(0)  # Restore original basis

    # Phase corrections to match complex amplitudes
    phi2 = np.angle(alpha) - np.angle(beta)  # Relative phase difference
    qc.crz(phi2, 0, 1)  # Controlled phase adjustment

    return qc
