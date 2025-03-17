"""Tests for HeisenbergSimulator module."""

import jax.numpy as jnp
import pytest
import stim
from typing import Any

from paulistringsquantumcircuitsimulations.circuit import Circuit, Gate
from paulistringsquantumcircuitsimulations.observable import Observable
from paulistringsquantumcircuitsimulations.heisenbergsimulator import heisenberg_simulate


def test_rx_rotation_expectation() -> None:
    theta = -3 * jnp.pi / 4
    circuit = Circuit(n=1)
    circuit.append(Gate(name="Ry", targets=[0], parameter=theta))
    hamiltonian = [
        Observable(coefficient=1.0, paulistring="X"),
        Observable(coefficient=1.0, paulistring="Z"),
    ]
    observables = heisenberg_simulate(circuit, hamiltonian)
    expectation = jnp.sum(jnp.array([obs.expectation() for obs in observables]))

    # After Rx rotation, expect cos(theta)
    assert jnp.allclose(expectation, -jnp.sqrt(2))
