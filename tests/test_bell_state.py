"""
Test suite for Bell state generation and verification.

This module tests the creation and measurement of entangled Bell states using
Hadamard and CNOT gates. The tests verify both the quantum state evolution
and the statistical distribution of measurement outcomes.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from experiments.bell_state import simulate_circuit


def test_bell_state_measurements():
    """
    Verify the measurement statistics of Bell state generation.

    The Bell state should produce approximately equal probabilities for
    ket(00) and ket(11) states, with near-zero probabilities for ket(01) and ket(10).
    """
    counts = simulate_circuit()

    # Verify we have exactly two measurement outcomes
    assert len(counts) == 2, "Bell state should produce only |00⟩ and |11⟩ outcomes"

    # Check for expected states
    assert "00" in counts, "ket(00) state missing"
    assert "11" in counts, "ket(11) state missing"

    # Verify approximate 50/50 distribution (within 5% tolerance)
    prob_00 = counts.get("00", 0) / 1000
    prob_11 = counts.get("11", 0) / 1000
    assert 0.45 < prob_00 < 0.55, "ket(00) probability out of expected range"
    assert 0.45 < prob_11 < 0.55, "ket(11) probability out of expected range"


def test_state_evolution():
    """Test the quantum state evolution through the circuit."""
    qc = QuantumCircuit(2)

    # Test initial state
    state = Statevector(qc)
    assert np.allclose(state, [1, 0, 0, 0]), "Initial state should be |00⟩"

    # Test after Hadamard on qubit 0
    qc.h(0)
    state = Statevector(qc)
    expected = [1 / np.sqrt(2), 1 / np.sqrt(2), 0, 0]
    assert np.allclose(
        state, expected, atol=1e-8
    ), f"Post-Hadamard state incorrect. Got {state}, expected {expected}"

    # Test after CNOT
    qc.cx(0, 1)
    state = Statevector(qc)
    expected_bell = [1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)]
    assert np.allclose(
        state, expected_bell, atol=1e-8
    ), f"Bell state not properly formed. Got {state}, expected {expected_bell}"
