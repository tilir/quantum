"""
Test suite for Bell state generation and verification.

This module tests the creation and measurement of entangled Bell states using
Hadamard and CNOT gates. The tests verify both the quantum state evolution
and the statistical distribution of measurement outcomes.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from experiments.qstates import print_opmatrix, print_statevector


def test_hadamard_gate_isolation():
    """Test Hadamard gate matrix in complete isolation"""
    qc = QuantumCircuit(1)  # Изолированная цепь с 1 кубитом
    qc.h(0)

    op = Operator(qc)
    expected_h = np.array([[1, 1], [1, -1]]) / np.sqrt(2)

    assert np.allclose(
        op.data, expected_h
    ), f"Hadamard gate matrix incorrect. Got:\n{op.data}\nExpected:\n{expected_h}"


# Basis states (little-endian: q1q0)
# ket(00) = [1,0,0,0]
# ket(01) = [0,1,0,0]
# ket(10) = [0,0,1,0]
# ket(11) = [0,0,0,1]


# CNOT operation (control = q0):
# ket(00) -> ket(00)
# ket(01) -> ket(11)
# ket(10) -> ket(10)
# ket(11) -> ket(01)
def test_cnot_gate_isolation():
    """Test CNOT gate matrix in complete isolation"""
    qc = QuantumCircuit(2)
    qc.cx(0, 1)

    op = Operator(qc)
    expected_cnot = np.array(
        [[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]], dtype=complex
    )

    assert np.allclose(
        op.data, expected_cnot
    ), f"CNOT gate matrix incorrect. Got:\n{op.data}\nExpected:\n{expected_cnot}"


def test_gate_sequence():
    """Test combined gate operations"""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)

    # Check join ,atrix
    op = Operator(qc)

    # Expected matrix for H(q0) -> CNOT(q0,q1)
    expected = np.array(
        [[1, 1, 0, 0], [0, 0, 1, -1], [0, 0, 1, 1], [1, -1, 0, 0]], dtype=complex
    ) / np.sqrt(2)

    assert np.allclose(
        op.data, expected
    ), f"Combined gate sequence incorrect. Got:\n{op.data}\nExpected:\n{expected}"


def test_print_functions(capsys):
    """
    Test the debug printing utilities.

    Verifies that statevector and operator matrix
    printing functions execute without errors.
    """
    qc = QuantumCircuit(2)
    print_statevector(qc, "Test state")
    print_opmatrix(qc, "Test operator")

    captured = capsys.readouterr()
    assert "Test state" in captured.out
    assert "Test operator" in captured.out
