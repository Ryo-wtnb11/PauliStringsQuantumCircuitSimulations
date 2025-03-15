from dataclasses import dataclass

import jax.numpy as jnp
import stim

from paulistringsquantumcircuitsimulations.circuit import Gate


@dataclass
class Observable:
    """Observable.

    Args:
        value (jnp.ndarray): The value of the observable.
        pauli_string (stim.PauliString): The PauliString representation of the observable.

    """

    coefficient: jnp.float64
    paulistring: stim.PauliString


def operator_evolution(
    n: int, observables: Observable | list[Observable], gate: Gate
) -> list[Observable]:
    """Operator evolution.

    Args:
        n (int): The number of qubits.
        observables (Observable | list[Observable]): The observable(s) to be evolved.
        gate (Gate): The gate to be applied.

    """
    if isinstance(observables, Observable):
        observables = [observables]
    new_observables: list[Observable] = []
    if gate.name in ["Rx", "Ry", "Rz", "T"]:
        paulistring = convert_paulistring(n, gate.name, gate.targets[0])
        for observable in observables:
            if observable.paulistring.commutes(paulistring):
                new_observables.append(
                    Observable(
                        coefficient=observable.coefficient,
                        paulistring=observable.paulistring,
                    ),
                )
            else:
                new_observables.extend(
                    [
                        Observable(
                            coefficient=observable.coefficient * jnp.cos(gate.parameter),
                            paulistring=observable.paulistring,
                        ),
                        Observable(
                            coefficient=observable.coefficient * jnp.sin(gate.parameter),
                            paulistring=1j * paulistring * observable.paulistring,
                        ),
                    ]
                )
    else:
        new_observables.extend(
            [
                Observable(
                    coefficient=observable.coefficient,
                    paulistring=observable.paulistring.before(
                        stim.CircuitInstruction(gate.name, gate.targets)
                    ),
                )
                for observable in observables
            ]
        )
    return new_observables


def expectation(observables: Observable | list[Observable]) -> jnp.ndarray:
    """Calculate expectation value of observables for |0...0⟩ state.

    Args:
        observables (Observable | list[Observable]): The observable(s) to be evaluated.

    Returns:
        jnp.ndarray: Expectation value. Can be differentiated using JAX.

    Example:
        >>> obs = Observable(coefficient=1.0, paulistring=stim.PauliString("Z"))
        >>> expectation(obs)  # ⟨0|Z|0⟩ = 1.0
        DeviceArray(1., dtype=float64)

    """
    if isinstance(observables, Observable):
        observables = [observables]

    total_expectation = jnp.zeros(1, dtype=jnp.float64)

    for obs in observables:
        xs, zs = obs.paulistring.to_numpy()
        local_expectation = jnp.ones(1, dtype=jnp.float64)  # JAXの配列として初期化

        for x, z in zip(xs, zs, strict=False):
            if (z and not x) or (not z and not x):  # Z component or Identity
                local_expectation = local_expectation * jnp.ones(1, dtype=jnp.float64)
            else:
                local_expectation = local_expectation * jnp.zeros(
                    1, dtype=jnp.float64
                )  # ⟨0|X|0⟩ = 0
        total_expectation = total_expectation + jnp.array([obs.coefficient]) * local_expectation

    return total_expectation


def convert_paulistring(n: int, gate: str, index: int) -> stim.PauliString:
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
    pauli_str = "".join(pauli_chars)

    return stim.PauliString(pauli_str)
