import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from jax import jit

from paulistringsquantumcircuitsimulations.circuit import Circuit
from paulistringsquantumcircuitsimulations.exceptions import InvalidParameterError, SystemSizeError
from paulistringsquantumcircuitsimulations.paulioperators import PauliOperators, PauliString


def evaluate_expectation_value_zero_state(
    pauli: PauliOperators,
    index: npt.NDArray[np.int64],
) -> jnp.ndarray:
    """Evaluate Pauli expectation value with respect to the |0> state.

    Args:
        pauli: PauliOperators
            The PauliOperators to evaluate the expectation value of.
        index: npt.NDArray[np.int64]
            The indices of the Pauli operators to evaluate the expectation value of.

    Returns:
        npt.NDArray[np.float64]: The expectation value of the Pauli operators.

    """
    return jnp.array(pauli.signs[index])


class HeisenbergSimulator:
    """A class for simulating Heisenberg models."""

    def __init__(
        self,
        circuit: Circuit,
        paulistrings: list[PauliString],
        n_qubits: int,
        coefficients: list[complex] | None = None,
        threshold: float = 0.0,
    ) -> None:
        if n_qubits != circuit.n_qubits:
            raise SystemSizeError(n_qubits, circuit.n_qubits)

        self.n_qubits = n_qubits

        circuit_paulistrings, circuit_signs = circuit.get_paulistrings()

        self.circuit_paulioperator_list = [
            PauliOperators.from_strings(
                paulistrings=[circuit_paulistrings[i]],
                signs=[circuit_signs[i]],
                n_qubits=self.n_qubits,
            )
            for i in range(len(circuit_paulistrings))
        ]

        observables_paulistrings, observables_signs = circuit.transform_paulistrings(
            paulistrings=paulistrings,
        )

        if coefficients is None:
            coefficients = [1.0] * len(observables_paulistrings)

        self.observables_paulioperators = PauliOperators.from_strings(
            paulistrings=observables_paulistrings,
            signs=observables_signs,
            coefficients=coefficients,
            n_qubits=n_qubits,
        )
        self.observables_paulioperators.order_paulis()

        self.threshold = threshold

    def run(self, parameters: jnp.ndarray) -> jnp.ndarray:
        """Run the Heisenberg simulator.

        Args:
            parameters: npt.NDArray[jnp.float64]
                The parameters of the circuit.

        Returns:
            npt.NDArray[jnp.float64]: The expectation value of the observables.

        """
        if parameters.shape[0] != len(self.circuit_paulioperator_list):
            raise InvalidParameterError(len(self.circuit_paulioperator_list), parameters.shape[0])

        for i in range(len(self.circuit_paulioperator_list)):
            self.apply_pauli_operator(
                circuit_paulioperator=self.circuit_paulioperator_list[i],
                parameter=parameters[i],
            )
        nonzero_pauli_indices = np.where(self.observables_paulioperators.ztype())[0]
        return jnp.real(
            jnp.sum(
                jnp.array(self.observables_paulioperators.coefficients[nonzero_pauli_indices])
                * evaluate_expectation_value_zero_state(
                    self.observables_paulioperators,
                    nonzero_pauli_indices,
                ),
            ),
        )

    def apply_pauli_operator(
        self,
        circuit_paulioperator: PauliOperators,
        parameter: jnp.ndarray,
    ) -> None:
        """Apply a Pauli operator to the circuit.

        Args:
            circuit_paulioperator: PauliOperators
                The Pauli operator to apply.
            parameter: npt.NDArray[jnp.float64]
                The parameter of the circuit.

        Returns:
            PauliOperators: The Pauli operator applied to the circuit.

        """
        anticommuting = np.where(
            self.observables_paulioperators.anticommutes(
                circuit_paulioperator,
            ),
        )[0]
        if len(anticommuting):
            new_paulis, new_pauli_indices, new_pauli_in_observables = self.multiply_operators(
                circuit_paulioperator,
                anticommuting,
            )
            coeffs_sin: jnp.ndarray = jnp.array(self.observables_paulioperators.coefficients[anticommuting])
            coeffs_sin = pmult(coeffs_sin, (1j) * jnp.sin(2 * parameter))

            new_coeffs: jnp.ndarray = update_coeffs(
                jnp.array(self.observables_paulioperators.coefficients),
                jnp.array(
                    self.observables_paulioperators.coefficients[
                        new_pauli_indices % self.observables_paulioperators.size()[0]
                    ],
                ),
                jnp.cos(2 * parameter),
                jnp.sin(2 * parameter),
                jnp.array(new_paulis.signs),
                jnp.array(
                    self.observables_paulioperators.signs[
                        new_pauli_indices % self.observables_paulioperators.size()[0]
                    ],
                ),
                anticommuting,
                new_pauli_in_observables,
            )
            self.observables_paulioperators.coefficients = np.array(new_coeffs)
            to_remove = a_lt_b(new_coeffs, self.threshold)
            np_to_remove: npt.NDArray[np.bool_] = np.array(to_remove).astype(np.bool_)
            np_to_remove = np_to_remove[anticommuting]
            if np.any(np_to_remove):
                self.observables_paulioperators.delete_pauli(anticommuting[np_to_remove])

            to_add: jnp.ndarray = a_gt_b_and_not_c(
                coeffs_sin,
                self.threshold,
                jnp.array(new_pauli_in_observables),
            )
            np_to_add: npt.NDArray[np.bool_] = np.array(to_add).astype(np.bool_)
            if np.any(np_to_add):
                self.add_new_paulis(new_paulis, coeffs_sin, np_to_add)

    def multiply_operators(
        self,
        operator: PauliOperators,
        anticommuting_indices: npt.NDArray[np.int64],
    ) -> tuple[PauliOperators, npt.NDArray[np.int64], npt.NDArray[np.bool_]]:
        new_pauli_operators = PauliOperators(
            self.observables_paulioperators.bits[anticommuting_indices, :],
            self.observables_paulioperators.signs[anticommuting_indices],
            self.observables_paulioperators.coefficients[anticommuting_indices],
            self.observables_paulioperators.n_qubits,
        )
        new_pauli_operators.compose_with(operator)
        new_pauli_indices = self.observables_paulioperators.find_pauli_indices(
            new_pauli_operators,
        )
        new_pauli_in_observables = self.observables_paulioperators.find_pauli(
            new_pauli_operators,
            new_pauli_indices,
        )

        return new_pauli_operators, new_pauli_indices, new_pauli_in_observables

    def add_new_paulis(
        self,
        new_paulis: PauliOperators,
        new_coeffs: jnp.ndarray,
        ind_to_add: npt.NDArray[np.bool_],
    ) -> None:
        """Add rows of new_paulis at indices ind_to_add to self.observable.

        These include Paulis that are above threshold and don't exist already in self.observable.

        Args:
            new_paulis: PauliOperators
                The Pauli operators to add.
            new_coeffs: jnp.ndarray
                The coefficients of the Pauli operators to add.
            ind_to_add: npt.NDArray[np.bool_]
                The indices of the Pauli operators to add.

        """
        paulis_to_add = PauliOperators(
            new_paulis.bits[ind_to_add, :],
            new_paulis.signs[ind_to_add],
            np.array(new_coeffs)[ind_to_add],
            new_paulis.n_qubits,
        )

        paulis_to_add.order_paulis()

        # Insert new Paulis and return new array of coefficients.
        self.observables_paulioperators.insert_pauli(paulis_to_add)


@jit
def pmult(
    a: jnp.ndarray,
    b: jnp.ndarray,
) -> jnp.ndarray:
    return a * b


def update_coeffs(  # noqa: PLR0913
    coeffs1: jnp.ndarray,
    coeffs2: jnp.ndarray,
    c: jnp.ndarray,
    s: jnp.ndarray,
    sign1: jnp.ndarray,
    sign2: jnp.ndarray,
    index1: npt.NDArray[np.int64],
    index_exists: npt.NDArray[np.bool_],
) -> jnp.ndarray:
    tmp = coeffs2 * (index_exists * (1j) * s * sign2 / sign1)
    return coeffs1.at[index1].set(jnp.take(coeffs1, index1) * c + tmp)


@jit
def a_lt_b(a: jnp.ndarray, b: float) -> jnp.ndarray:
    """Compare absolute values of vector elements with a scalar.

    Args:
        a: Vector input array
        b: Scalar threshold value

    Returns:
        Boolean array where True indicates |a[i]| < b

    """
    return jnp.abs(a) < b


@jit
def a_gt_b_and_not_c(a: jnp.ndarray, b: float, c: jnp.ndarray) -> jnp.ndarray:
    return (jnp.abs(a) >= b) & ~c
