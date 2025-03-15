"""Tests for Observable module."""

import jax.numpy as jnp
import pytest
import stim
from typing import Any

from paulistringsquantumcircuitsimulations import Gate, Observable, operator_evolution, expectation


def test_observable_creation() -> None:
    """Test Observable instance creation."""
    obs = Observable(coefficient=1.0, paulistring=stim.PauliString("Z"))
    assert obs.coefficient == 1.0
    assert str(obs.paulistring) == "+Z"

def test_rx_rotation_expectation() -> None:
    """Test expectation value after Rx rotation."""
    theta = jnp.pi / 4
    obs = Observable(coefficient=1.0, paulistring=stim.PauliString("Z"))
    gate = Gate(name="Rx", targets=[0], parameter=theta)
    evolved_obs = operator_evolution(1, obs, gate)
    exp_val = expectation(evolved_obs)

    # After Rx rotation, expect cos(theta)
    assert jnp.allclose(exp_val, jnp.cos(theta))
