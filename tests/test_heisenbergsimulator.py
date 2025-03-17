"""Tests for Observable module."""

import jax.numpy as jnp
import pytest
import stim
from typing import Any

from paulistringsquantumcircuitsimulations import Gate, Circuit, Observable, heisenberg_simulate

def test_observable_creation() -> None:
    """Test Observable instance creation."""
    obs = Observable(coefficient=1.0, paulistring=stim.PauliString("Z"))
    assert obs.coefficient == 1.0
    assert str(obs.paulistring) == "+Z"

def test_rx_rotation_expectation() -> None:
    theta = - 3 * jnp.pi / 4
    circuit = Circuit(n=1)
    circuit.append(Gate(name="Ry", targets=[0], parameter=theta))
    hamiltonian = [
        Observable(coefficient=1.0, paulistring=stim.PauliString("X")),
        Observable(coefficient=1.0, paulistring=stim.PauliString("Z")),
    ]
    observables = heisenberg_simulate(circuit, hamiltonian)
    expectation = jnp.sum(jnp.array([obs.expectation() for obs in observables]))

    # After Rx rotation, expect cos(theta)
    assert jnp.allclose(expectation, -jnp.sqrt(2))
