#!/usr/bin/env python3

"""
Tests for Quantum Zeno Effect demonstration
"""

import pytest
import numpy as np
import random
from unittest.mock import patch
from qiskit.quantum_info import Statevector
from experiments.qzeno import (qzeno, 
                              measurement_projectors,
                              measure_single_qzeno,
                              THETA_RAD, 
                              NUM_MEASUREMENTS, 
                              NUM_ATTEMPTS)

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
    P_v, P_w = measurement_projectors(theta)
    
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
    with patch('random.random', return_value=0.1):
        state = Statevector([1, 0])
        final_state = measure_single_qzeno(state, theta, 3)
        
        # Правильный расчёт ожидаемого состояния
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        norm = np.sqrt(cos_theta**6 + (cos_theta**2 * sin_theta)**2)
        expected_0 = cos_theta**3 / norm
        expected_1 = cos_theta**2 * sin_theta / norm
        
        # Проверяем эквивалентность с учётом глобальной фазы
        assert np.allclose(np.abs(final_state.data), np.abs([expected_0, expected_1]), atol=1e-7), (
            f"State magnitudes don't match. Expected {[expected_0, expected_1]}, got {final_state.data}"
        )
    
    # Test case 2: Alternating P_v and P_w measurements
    with patch('random.random', side_effect=[0.1, 0.9, 0.1]):
        state = Statevector([1, 0])
        final_state = measure_single_qzeno(state, theta, 3)
        
        # Практический расчёт:
        state = Statevector([1, 0])
        P_v, P_w = measurement_projectors(theta)
        
        # First P_v
        prob_v = np.abs(state.data.conj() @ P_v @ state.data)
        state = Statevector((P_v @ state.data)/np.sqrt(prob_v))
        
        # Then P_w
        prob_w = np.abs(state.data.conj() @ P_w @ state.data)
        state = Statevector((P_w @ state.data)/np.sqrt(prob_w))
        
        # Then P_v again
        prob_v = np.abs(state.data.conj() @ P_v @ state.data)
        expected_state = Statevector((P_v @ state.data)/np.sqrt(prob_v))
        
        # Проверяем эквивалентность с учётом глобальной фазы
        assert np.allclose(np.abs(final_state.data), np.abs(expected_state.data), atol=1e-7), (
            f"State magnitudes don't match. Expected {expected_state.data}, got {final_state.data}"
        )

def test_qzeno_basic_properties(fixed_seed):
    """Test basic properties of qzeno experiment results"""
    results = qzeno(num_attempts=100)
    
    # Should return exactly two counts
    assert len(results) == 2, "Results should contain counts for |0⟩ and |1⟩"
    
    # Total counts should equal num_attempts
    assert sum(results) == 100, "Total counts should match number of attempts"
    
    # All counts should be non-negative
    assert all(count >= 0 for count in results), "Counts should be non-negative"

def test_qzeno_zeno_effect_strong(fixed_seed):
    """Test strong Zeno effect with small rotation angles"""
    # Small angle (1 degree) - strong Zeno effect
    results = qzeno(theta_rad=np.radians(1), num_attempts=100)
    prob_0 = results[0] / 100
    assert prob_0 > 0.95, "With small angles, |0⟩ probability should be >95%"

def test_qzeno_no_zeno_effect(fixed_seed):
    """Test case without Zeno effect (single large rotation)"""
    # Single 45 degree measurement - no Zeno effect
    results = qzeno(theta_rad=np.radians(45), num_measurements=1, num_attempts=100)
    prob_0 = results[0] / 100
    assert 0.4 < prob_0 < 0.6, "With single 45° measurement, ket(0) probability should be ~50%"

@pytest.mark.parametrize("theta_deg, min_prob_0", [
    (0.5, 0.98),   # Very small angle - strong Zeno
    (2, 0.9),      # Default case
    (10, 0.7),     # Moderate angle
    (30, 0.4)      # Larger angle - weaker Zeno
])
def test_qzeno_angle_dependence(theta_deg, min_prob_0, fixed_seed):
    """Test Zeno effect dependence on measurement angle"""
    results = qzeno(
        theta_rad=np.radians(theta_deg),
        num_attempts=1000  # More attempts for better statistics
    )
    prob_0 = results[0] / sum(results)
    assert prob_0 > min_prob_0, f"With {theta_deg}° rotations, |0⟩ probability should be >{min_prob_0}"

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