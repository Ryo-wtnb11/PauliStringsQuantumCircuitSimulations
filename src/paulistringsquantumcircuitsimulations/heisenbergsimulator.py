from typing import Self

import jax.numpy as jnp
import stim

from paulistringsquantumcircuitsimulations.circuit import Circuit, Gate
from paulistringsquantumcircuitsimulations.exceptions import ObservableLengthError


class Observable:
    """Observable.

    Args:
        value (jnp.ndarray): The value of the observable.
        pauli_string (stim.PauliString): The PauliString representation of the observable.

    """

    def __init__(self, coefficient: jnp.float64, paulistring: stim.PauliString) -> None:
        self.coefficient: jnp.float64 = coefficient
        self.paulistring: stim.PauliString = paulistring

    def commutes(self, other: Self | stim.PauliString) -> bool:
        """Check if this observable commutes with another.

        Args:
            other: Another Observable instance to check commutation with

        Returns:
            bool: True if the observables commute, False otherwise

        """
        result: bool = False
        if isinstance(other, type(self)):
            result = self.paulistring.commutes(other.paulistring)
        else:
            result = self.paulistring.commutes(other)
        return result

    def expectation(self) -> jnp.ndarray:
        """Calculate expectation value of observables for |0...0⟩ state.

        Returns:
            jnp.ndarray: Expectation value. Can be differentiated using JAX.

        Example:
            >>> obs = Observable(coefficient=1.0, paulistring=stim.PauliString("Z"))
            >>> obs.expectation()  # ⟨0|Z|0⟩ = 1.0
            DeviceArray(1., dtype=float64)

        """
        xs, zs = self.paulistring.to_numpy()
        xs = jnp.array(xs, dtype=jnp.bool)
        zs = jnp.array(zs, dtype=jnp.bool)

        if jnp.any(xs):
            return jnp.array(0.0, dtype=jnp.float64)
        return jnp.array(jnp.real(self.coefficient * self.paulistring.sign), dtype=jnp.float64)


class HeisenbergSimulator:
    """Heisenberg Simulator.

    Args:
        circuit (Circuit): The circuit to be simulated.
        observables (list[Observable]): The observables to be simulated.
        threshold (float): The threshold for the coefficient of the observables.

    """

    def __init__(self, circuit: Circuit, observables: list[Observable], threshold: float = 1e-8) -> None:
        self.circuit = circuit
        self.observables = observables
        self.threshold = threshold

        # Check if the observable length exceeds the circuit size
        for observable in self.observables:
            if len(observable.paulistring) > self.circuit.n:
                raise ObservableLengthError(len(observable.paulistring), self.circuit.n)

    def simulate(self, threshold: float | None = None) -> list[Observable]:
        """Simulate the circuit.

        Args:
            inplace: If True, modify the observables in-place. If False, return a new list.
            threshold: Optional threshold for filtering observables. If None, use the instance threshold.

        Returns:
            list[Observable] | None: The observables after the circuit is applied if inplace=False.
            None: If inplace=True and the observables are modified in-place.

        """
        evolved_observables = self.observables
        current_threshold = threshold if threshold is not None else self.threshold

        for gate in reversed(self.circuit.instructions):
            evolved_observables = _operator_evolution(self.circuit.n, evolved_observables, gate)

            # Remove observables with coefficients below the threshold
            evolved_observables = [
                observable
                for observable in evolved_observables
                if abs(observable.coefficient) > current_threshold
            ]

        return evolved_observables


def _operator_evolution(
    n: int,
    observables: Observable | list[Observable],
    gate: Gate,
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
    if gate.name in ["Rx", "Ry", "Rz"]:
        paulistring = convert_paulistring(n, gate.name, gate.targets[0])
        for observable in observables:
            if observable.commutes(paulistring):
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
                    ],
                )
    else:
        new_observables.extend(
            [
                Observable(
                    coefficient=observable.coefficient,
                    paulistring=observable.paulistring.before(
                        stim.CircuitInstruction(gate.name, gate.targets),
                    ),
                )
                for observable in observables
            ],
        )

    # TODO: Combine observables with the same Pauli string to reduce the number of terms.
    return new_observables


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
