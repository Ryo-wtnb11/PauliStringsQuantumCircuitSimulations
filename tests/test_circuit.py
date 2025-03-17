"""Tests for Circuit module."""

import jax.numpy as jnp
import pytest
import stim
from typing import Any

from paulistringsquantumcircuitsimulations.circuit import Circuit, Gate


def test_circuit_creation() -> None:
    """Test Observable instance creation."""
    circuit = Circuit(n=2)
    assert circuit.n == 2
    assert len(circuit.instructions) == 0

    circuit.append(Gate(name="Rx", targets=[0], parameter=jnp.pi / 2))
    assert len(circuit.instructions) == 1
    assert circuit.instructions[0].name == "Rx"
    assert circuit.instructions[0].targets == [0]
    assert circuit.instructions[0].parameter == jnp.pi / 2
