#!/usr/bin/env python3

"""
Tests for Quantum Zeno Effect demonstration
"""

import random
from unittest.mock import patch

import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from experiments.qzeno import (
    THETA_RAD,
    qzeno_measurements,
    measurement_projectors,
    qzeno,
)


@pytest.fixture
def fixed_seed():
    """Fixture to set fixed random seed for deterministic tests"""
    seed = 42
    np.random.seed(seed)
    random.seed(seed)
    yield
    # Cleanup - reset random state
    np.random.seed(None)
    random.seed(None)


def test_measurement_projectors_basic_properties():
    """Test that measurement projectors satisfy required mathematical properties"""
    theta = np.radians(30)
    v = np.array([np.cos(theta), np.sin(theta)])
    w = np.array([-np.sin(theta), np.cos(theta)])

    P_v, P_w = measurement_projectors(v, w)

    # 1. Test projectors are Hermitian
    assert np.allclose(P_v, P_v.conj().T), "P_v should be Hermitian"
    assert np.allclose(P_w, P_w.conj().T), "P_w should be Hermitian"

    # 2. Test orthogonality
    assert np.allclose(P_v @ P_w, np.zeros((2, 2))), "Projectors should be orthogonal"

    # 3. Test completeness
    assert np.allclose(P_v + P_w, np.eye(2)), "Projectors should sum to identity"


def test_measure_single_qzeno_deterministic():
    """Test measurement sequence with forced measurement outcomes"""
    theta = THETA_RAD  # ~2 degrees

    # Test case 1: All measurements result in P_v
    with patch("numpy.random.random", return_value=0.0):  # 20 zeroes in a row
        state = np.array([1, 0])
        final_state = qzeno_measurements(state, theta, 20)
        expected_state = [np.cos(20*theta), np.sin(20*theta)] 

        # Check for equivalence up to global phase
        assert np.allclose(
            np.abs(final_state), np.abs(expected_state), atol=1e-7
        ), f"State magnitudes don't match. Expected {expected_state}, got {final_state}"

    # Test case 2: All measurements result in P_w
    with patch("numpy.random.random", return_value=1.0):  # 20 ones in a row
        state = np.array([1, 0])
        final_state = qzeno_measurements(state, theta, 20)
        expected_state = [np.sin(20*theta), -np.cos(20*theta)] 

        # Check for equivalence up to global phase
        assert np.allclose(
            np.abs(final_state), np.abs(expected_state), atol=1e-2
        ), f"State magnitudes don't match. Expected {expected_state}, got {final_state}"


def test_qzeno_basic_properties(fixed_seed):
    """Test basic properties of qzeno experiment results"""
    results = qzeno(num_attempts=100)

    # Should return exactly two counts
    assert len(results) == 2, "Results should contain counts for ket(0) and ket(1)"

    # Total counts should equal num_attempts
    assert sum(results) == 100, "Total counts should match number of attempts"

    # All counts should be non-negative
    assert all(count >= 0 for count in results), "Counts should be non-negative"


def test_qzeno_zeno_effect_strong(fixed_seed):
    """Test strong Zeno effect with small rotation angles"""
    # Small angle (1 degree) - strong Zeno effect
    results = qzeno(theta_rad=np.radians(1), num_attempts=100)
    prob_1 = results[1] / 100
    assert prob_1 > 0.95, "With small angles, ket(0) probability should be >95%"


def test_qzeno_no_zeno_effect(fixed_seed):
    """Test case without Zeno effect (single large rotation)"""
    # Single 45 degree measurement - no Zeno effect
    results = qzeno(theta_rad=np.radians(45), num_measurements=1, num_attempts=100)
    prob_1 = results[1] / 100
    assert (
        0.4 < prob_1 < 0.6
    ), "With single 45° measurement, ket(0) probability should be ~50%"


@pytest.mark.parametrize(
    "theta_deg, min_prob_1",
    [
        (0.5, 0.98),  # Very small angle - strong Zeno
        (2, 0.9),  # Default case
        (10, 0.7),  # Moderate angle
        (30, 0.4),  # Larger angle - weaker Zeno
    ],
)
def test_qzeno_angle_dependence(theta_deg, min_prob_1, fixed_seed):
    """Test Zeno effect dependence on measurement angle"""
    results = qzeno(
        theta_rad=np.radians(theta_deg),
        num_attempts=1000,  # More attempts for better statistics
    )
    prob_1 = results[1] / sum(results)
    assert (
        prob_1 > min_prob_1
    ), f"With {theta_deg}° rotations, ket(0) probability should be >{min_prob_1}"


def test_qzeno_reproducibility():
    """Test that results are reproducible with same random seed"""
    results1 = qzeno(num_attempts=100, random_seed=42)
    results2 = qzeno(num_attempts=100, random_seed=42)
    assert results1 == results2, "Results should be identical with same random seed"


def test_qzeno_edge_cases():
    """Test edge cases and error handling"""
    # Zero angle should raise error
    with pytest.raises(ValueError, match="theta_rad must be positive"):
        qzeno(theta_rad=0)

    # Negative angle should raise error
    with pytest.raises(ValueError, match="theta_rad must be positive"):
        qzeno(theta_rad=-0.1)

    # Zero attempts should return zero counts
    with pytest.raises(ValueError, match="num_attempts must be positive"):
        qzeno(num_attempts=0)
