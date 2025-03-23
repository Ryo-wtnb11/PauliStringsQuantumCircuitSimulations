import numpy as np
import jax.numpy as jnp
import stim

from paulistringsquantumcircuitsimulations.circuit import Circuit, Gate
from paulistringsquantumcircuitsimulations.heisenbergsimulator import HeisenbergSimulator

def expectation(paulistrings: stim.PauliString, coefficients: jnp.ndarray) -> jnp.ndarray:
    xs, zs = paulistrings.to_numpy()
    xs = jnp.array(xs, dtype=jnp.bool_)
    zs = jnp.array(zs, dtype=jnp.bool_)

    if jnp.any(xs):
        return jnp.array(0.0, dtype=jnp.float64)
    return jnp.array(jnp.real(paulistrings.sign * coefficients), dtype=jnp.float64)


def test_heisenberg_simulator() -> None:
    n = 1
    circuit = Circuit(n_qubits=n)
    circuit.append(Gate(name="Ry", targets=[0]))

    for c in [2, -2, 4, -4, 8, -8, 8 / 5, -8 / 5]:
        paulistrings = ["X", "Z"]
        simulator = HeisenbergSimulator.init_circuit(
            circuit=circuit,
            paulistrings=paulistrings,
            n_qubits=n,
        )
        parameters = jnp.pi / c

        theta: jnp.ndarray = jnp.array([parameters], dtype=jnp.float64)
        exp = simulator.run_circuit(theta)

        stim_x = stim.PauliString("X")
        stim_z = stim.PauliString("Z")

        exp_x = jnp.array(0.0, dtype=jnp.float64)
        if stim_x.commutes(stim.PauliString("Y")):
            exp_x = expectation(stim_x, jnp.array(1.0, dtype=jnp.float64))
        else:
            stim_yx = stim.PauliString("Y") * stim_x
            exp_x = expectation(stim_x, jnp.cos(parameters * 2)) + expectation(
                stim_yx,
                (1.0j) * jnp.sin(parameters * 2),
            )

        exp_z = jnp.array(0.0, dtype=jnp.float64)
        if stim_z.commutes(stim.PauliString("Y")):
            exp_z = expectation(stim_z, jnp.array(1.0, dtype=jnp.float64))
        else:
            stim_yz = stim.PauliString("Y") * stim_z
            exp_z = expectation(stim_z, jnp.cos(parameters * 2)) + expectation(
                stim_yz,
                (1.0j) * jnp.sin(parameters * 2),
            )

        exp_ = exp_x + exp_z

        assert np.isclose(np.array(exp), np.array(exp_))

def test_heisenberg_simulator_including_clifford_gates() -> None:
    n = 1
    circuit = Circuit(n_qubits=n)
    circuit.append(Gate(name="H", targets=[0]))
    circuit.append(Gate(name="Rz", targets=[0]))
    circuit.append(Gate(name="H", targets=[0]))

    paulistrings = ["X", "Z"]
    simulator = HeisenbergSimulator.init_circuit(
        circuit=circuit,
        paulistrings=paulistrings,
        n_qubits=n,
    )

    parameters = jnp.pi / 8
    theta: jnp.ndarray = jnp.array([parameters], dtype=jnp.float64)
    exp = simulator.run_circuit(theta)

    stim_x = stim.PauliString("X")
    stim_z = stim.PauliString("Z")

    stim_x = stim_x.before(stim.CircuitInstruction("H", [0]))
    stim_z = stim_z.before(stim.CircuitInstruction("H", [0]))

    exp_x = jnp.array(0.0, dtype=jnp.float64)
    if stim_x.commutes(stim.PauliString("Z")):
        stim_x = stim_x.before(stim.CircuitInstruction("H", [0]))
        exp_x = expectation(stim_x, jnp.array(1.0, dtype=jnp.float64))
    else:
        stim_xz = stim.PauliString("Z") * stim_x
        stim_x = stim_x.before(stim.CircuitInstruction("H", [0]))
        stim_xz = stim_xz.before(stim.CircuitInstruction("H", [0]))
        exp_x = expectation(stim_x, jnp.cos(parameters * 2)) + expectation(
            stim_xz,
            (1.0j) * jnp.sin(parameters * 2),
        )

    exp_z = jnp.array(0.0, dtype=jnp.float64)
    if stim_z.commutes(stim.PauliString("Z")):
        stim_z = stim_z.before(stim.CircuitInstruction("H", [0]))
        exp_z = expectation(stim_z, jnp.array(1.0, dtype=jnp.float64))
    else:
        stim_zz = stim.PauliString("Z") * stim_z
        stim_z = stim_z.before(stim.CircuitInstruction("H", [0]))
        stim_zz = stim_zz.before(stim.CircuitInstruction("H", [0]))
        exp_z = expectation(stim_z, jnp.cos(parameters * 2)) + expectation(
            stim_zz,
            (1.0j) * jnp.sin(parameters * 2),
        )

    exp_ = exp_x + exp_z

    assert jnp.isclose(jnp.array(exp), jnp.array(exp_))