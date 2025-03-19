"""Tests for Circuit module."""

import jax.numpy as jnp
import pytest
import stim
from typing import Any

from paulistringsquantumcircuitsimulations.circuit import Circuit, Gate


def test_circuit_creation() -> None:
    """Test circuit instance creation."""
    n = 2
    circuit = Circuit(n_qubits=n)
    gates = [
        Gate(name="H", targets=[0]),
        Gate(name="H", targets=[1]),
        Gate(name="Rz", targets=[0]),
        Gate(name="CNOT", targets=[0, 1]),
        Gate(name="Rx", targets=[1]),
        Gate(name="H", targets=[0]),
    ]
    for gate in gates:
        circuit.append(gate)

    paulistrings, signs = circuit.get_paulistrings()
    stim_z = stim.PauliString("Z_")
    stim_z = stim_z.before(stim.CircuitInstruction("H", [0]))
    stim_z = stim_z.before(stim.CircuitInstruction("H", [1]))


    stim_x = stim.PauliString("_X")
    stim_x = stim_x.before(stim.CircuitInstruction("CNOT", [0, 1]))
    stim_x = stim_x.before(stim.CircuitInstruction("H", [0]))
    stim_x = stim_x.before(stim.CircuitInstruction("H", [1]))

    assert paulistrings[0] == str(stim_z)[-n:]
    assert paulistrings[1] == str(stim_x)[-n:]

    op = ["X_", "_Y"]
    paulistrings, signs = circuit.transform_paulistrings(op)

    stim_x = stim.PauliString("X_")
    stim_x = stim_x.before(stim.CircuitInstruction("H", [0]))
    stim_x = stim_x.before(stim.CircuitInstruction("CNOT", [0, 1]))
    stim_x = stim_x.before(stim.CircuitInstruction("H", [1]))
    stim_x = stim_x.before(stim.CircuitInstruction("H", [0]))

    stim_y = stim.PauliString("_Y")
    stim_y = stim_y.before(stim.CircuitInstruction("H", [0]))
    stim_y = stim_y.before(stim.CircuitInstruction("CNOT", [0, 1]))
    stim_y = stim_y.before(stim.CircuitInstruction("H", [1]))
    stim_y = stim_y.before(stim.CircuitInstruction("H", [0]))

    assert paulistrings[0] == str(stim_x)[-n:]
    assert paulistrings[1] == str(stim_y)[-n:]
