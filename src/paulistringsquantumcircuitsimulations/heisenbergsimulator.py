import jax.numpy as jnp
import stim

from paulistringsquantumcircuitsimulations.circuit import Circuit
from paulistringsquantumcircuitsimulations.exceptions import ObservableLengthError
from paulistringsquantumcircuitsimulations.observable import Observable, PauliString


def heisenberg_simulate(
    circuit: Circuit,
    observables: list[Observable],
    threshold: float = 1e-8,
    xy_weight: int = -1,
) -> list[Observable]:
    """Simulate the circuit.

    Args:
        observables (dict[PauliString, jnp.complex128]): The observables to be simulated.
        circuit (Circuit): The circuit to be simulated.
        threshold: Optional threshold for filtering observables. If None, use the instance threshold.
        xy_weight: Optional weight for the XY plane. If -1, use the circuit length.

    Returns:
        dict[PauliString, jnp.complex128]: The observables after the circuit is applied.

    """
    for gate in reversed(circuit.instructions):
        observables_dict: dict[PauliString, jnp.complex128] = {}
        for observable in observables:
            if len(observable.paulistring) > circuit.n:
                raise ObservableLengthError(len(observable.paulistring), circuit.n)

            gate_paulistring = _convert_paulistring(circuit.n, gate.name, gate.targets[0])
            if gate.name in ["Rx", "Ry", "Rz", "T"]:
                if stim.PauliString(observable.paulistring).commutes(stim.PauliString(gate_paulistring)):
                    observables_dict[observable.paulistring] = (
                        observables_dict.get(observable.paulistring, jnp.complex128(0.0))
                        + observable.coefficient
                    )
                else:
                    observables_dict[observable.paulistring] = (
                        observables_dict.get(observable.paulistring, jnp.complex128(0.0))
                        + jnp.cos(gate.parameter) * observable.coefficient
                    )

                    po_stim_paulistring = stim.PauliString(gate_paulistring) * stim.PauliString(
                        observable.paulistring,
                    )
                    paulistring_ = str(po_stim_paulistring)[-circuit.n :]
                    observables_dict[paulistring_] = (
                        observables_dict.get(paulistring_, jnp.complex128(0.0))
                        + 1.0j * po_stim_paulistring.sign * jnp.sin(gate.parameter) * observable.coefficient
                    )
            else:
                o_stim_paulistring = stim.PauliString(observable.paulistring)
                pop_stim_paulistring = o_stim_paulistring.before(
                    stim.CircuitInstruction(gate.name, gate.targets),
                )
                paulistring_ = str(pop_stim_paulistring)[-circuit.n :]
                observables_dict[paulistring_] = (
                    observables_dict.get(paulistring_, jnp.complex128(0.0)) + observable.coefficient
                )

        # Remove observables with coefficients below the threshold
        xy_weight = circuit.n if xy_weight == -1 else xy_weight
        new_observables: list[Observable] = [
            Observable(coefficient=coefficient, paulistring=paulistring)
            for paulistring, coefficient in observables_dict.items()
            if jnp.abs(coefficient) > threshold
            and paulistring.count("X") + paulistring.count("Y") <= xy_weight
        ]
        observables = new_observables

    return observables


def _convert_paulistring(n: int, gate: str, index: int) -> PauliString:
    """Convert a gate to a PauliString.

    Args:
        n (int): The number of qubits.
        gate (str): The gate to be converted.
        index (int): The index of the qubit to be converted.

    Returns:
        stim.PauliString: The PauliString representation of the gate.

    """
    gate_symbols = {"Rx": "X", "Ry": "Y", "Rz": "Z", "T": "Z"}

    pauli_chars: list[str] = ["_"] * n
    pauli_chars[index] = gate_symbols.get(gate, "_")
    pauli_str: PauliString = "".join(pauli_chars)

    return pauli_str
