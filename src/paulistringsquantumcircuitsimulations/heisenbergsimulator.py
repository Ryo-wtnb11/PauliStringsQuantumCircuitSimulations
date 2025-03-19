import numpy as np
import numpy.typing as npt

from paulistringsquantumcircuitsimulations.circuit import Circuit
from paulistringsquantumcircuitsimulations.exceptions import SystemSizeError
from paulistringsquantumcircuitsimulations.paulioperators import PauliOperators


def evaluate_expectation_value_zero_state(
    pauli: PauliOperators,
    index: npt.NDArray[np.int64],
) -> npt.NDArray[np.float64]:
    """Evaluate Pauli expectation value with respect to the |0> state.

    Args:
        pauli: PauliOperators
            The PauliOperators to evaluate the expectation value of.
        index: npt.NDArray[np.int64]
            The indices of the Pauli operators to evaluate the expectation value of.

    Returns:
        npt.NDArray[np.complex128]: The expectation value of the Pauli operators.

    """
    return np.real(pauli.signs[index])


class HeisenbergSimulator:
    """A class for simulating Heisenberg models."""

    def __init__(self, observables: PauliOperators, circuit: Circuit) -> None:
        if observables.n_qubits != circuit.n_qubits:
            raise SystemSizeError(observables.n_qubits, circuit.n_qubits)
        self.observables = observables
        self.circuit = circuit
        self.n_qubits = observables.n_qubits
