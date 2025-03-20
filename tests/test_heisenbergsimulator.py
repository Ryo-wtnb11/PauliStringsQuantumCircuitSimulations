import numpy as np
import jax.numpy as jnp
import stim

from paulistringsquantumcircuitsimulations.circuit import Circuit, Gate
from paulistringsquantumcircuitsimulations.heisenbergsimulator import HeisenbergSimulator
from paulistringsquantumcircuitsimulations.paulioperators import PauliOperators

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
        simulator = HeisenbergSimulator(
            circuit=circuit,
            paulistrings=paulistrings,
            n_qubits=n,
        )
        parameters = jnp.pi / c

        theta: jnp.ndarray = jnp.array([parameters], dtype=jnp.float64)
        exp = simulator.run(theta)

        stim_x = stim.PauliString("X")
        stim_z = stim.PauliString("Z")

        stim_yx = stim.PauliString("Y") * stim_x
        stim_yz = stim.PauliString("Y") * stim_z

        exp_ = (
            expectation(stim_x, jnp.cos(parameters * 2))
            + expectation(
                stim_yx,
                (1.0j) * jnp.sin(parameters * 2),
            )
            + expectation(stim_z, jnp.cos(parameters * 2))
            + expectation(stim_yz, (1.0j) * jnp.sin(parameters * 2))
        )
        assert np.isclose(np.array(exp), np.array(exp_))