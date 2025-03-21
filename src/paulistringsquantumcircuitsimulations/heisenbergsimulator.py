import copy

import jax
import jax.numpy as jnp
from jaxtyping import Bool, Complex128, Float64, UInt64

from paulistringsquantumcircuitsimulations.circuit import Circuit
from paulistringsquantumcircuitsimulations.exceptions import InvalidParameterError, SystemSizeError
from paulistringsquantumcircuitsimulations.paulioperators import (
    PauliString,
    anticommutes,
    compose_with,
    delete_paulioperators,
    find_paulioperators,
    find_paulioperators_indices,
    insert_paulioperators,
    order_paulioperators,
    paulioperators_from_strings,
    ztype,
)


def evaluate_expectation_value_zero_state(
    signs: Complex128[jnp.ndarray, " n_op"],
    index: UInt64[jnp.ndarray, " n"],
) -> Complex128[jnp.ndarray, " n"]:
    """Evaluate Pauli expectation value with respect to the |0> state.

    Args:
        signs: Complex128[jnp.ndarray, " n_op"]
            The signs of the Pauli operators to evaluate the expectation value of.
        index: UInt64[jnp.ndarray, " n"]
            The indices of the Pauli operators to evaluate the expectation value of.

    Returns:
        Complex128[jnp.ndarray, " n"]: The expectation value of the Pauli operators.

    """
    return signs[index]


class HeisenbergSimulator:
    """A class for simulating Heisenberg models."""

    def __init__(
        self,
        circuit: Circuit,
        paulistrings: list[PauliString],
        n_qubits: int,
        coefficients: Complex128[jnp.ndarray, " n_op"] | list[complex] | None = None,
        threshold: float = 0.0,
    ) -> None:
        if n_qubits != circuit.n_qubits:
            raise SystemSizeError(n_qubits, circuit.n_qubits)

        self.n_qubits = n_qubits

        circuit_paulistrings, circuit_signs = circuit.get_paulistrings()

        self.circuit_bit_list: list[UInt64[jnp.ndarray, "1 n_qubits"]] = []
        self.circuit_sign_list: list[Complex128[jnp.ndarray, "1 n_qubits"]] = []
        for i in range(len(circuit_paulistrings)):
            circuit_bit, circuit_sign, _ = paulioperators_from_strings(
                paulistrings=[circuit_paulistrings[i]],
                signs=jnp.array([circuit_signs[i]], dtype=jnp.complex128),
                n_qubits=self.n_qubits,
            )
            self.circuit_bit_list.append(circuit_bit)
            self.circuit_sign_list.append(circuit_sign)

        paulistrings, signs = circuit.transform_paulistrings(
            paulistrings=paulistrings,
        )

        observables_bits, observables_signs, observables_coefficients = paulioperators_from_strings(
            paulistrings=paulistrings,
            signs=signs,
            coefficients=coefficients,
            n_qubits=n_qubits,
        )
        self.observables_bits, self.observables_signs, self.observables_coefficients = order_paulioperators(
            bits=observables_bits,
            signs=observables_signs,
            coefficients=observables_coefficients,
        )

        self.threshold = threshold

    def run(
        self,
        parameters: Float64[jnp.ndarray, " n_circuit_parameters"],
    ) -> jnp.float64:
        """Run the Heisenberg simulator.

        Args:
            parameters: Float64[jnp.ndarray, " n_circuit_parameters"]
                The parameters of the circuit.

        Returns:
            jnp.float64: The expectation value of the observables.

        """
        if parameters.shape[0] != len(self.circuit_bit_list):
            raise InvalidParameterError(len(self.circuit_bit_list), parameters.shape[0])

        observables_bits, observables_signs, observables_coefficients = (
            copy.deepcopy(self.observables_bits),
            copy.deepcopy(self.observables_signs),
            copy.deepcopy(self.observables_coefficients),
        )
        for i in range(parameters.shape[0]):
            observables_bits, observables_signs, observables_coefficients = self.apply_pauli_operator(
                observables_bits=observables_bits,
                circuit_bit=self.circuit_bit_list[i],
                observables_signs=observables_signs,
                circuit_sign=self.circuit_sign_list[i],
                observables_coefficients=observables_coefficients,
                parameter=parameters[i],
            )
        nonzero_pauli_indices = jnp.where(ztype(self.observables_bits, self.n_qubits))[0]
        return jnp.real(
            jnp.sum(
                observables_coefficients[nonzero_pauli_indices]
                * evaluate_expectation_value_zero_state(
                    observables_signs,
                    nonzero_pauli_indices,
                ),
            ),
        )

    def apply_pauli_operator(
        self,
        observables_bits: UInt64[jnp.ndarray, " n_op n_qubits"],
        circuit_bit: UInt64[jnp.ndarray, "1 n_qubits"],
        observables_signs: Complex128[jnp.ndarray, " n_op"],
        circuit_sign: Complex128[jnp.ndarray, "1 n_qubits"],
        observables_coefficients: Complex128[jnp.ndarray, " n_op"],
        parameter: jnp.float64,
    ) -> tuple[
        UInt64[jnp.ndarray, "n_op+n n_qubits"],
        Complex128[jnp.ndarray, " n_op+n"],
        Complex128[jnp.ndarray, " n_op+n"],
    ]:
        """Apply a Pauli operator to the circuit.

        Args:
            observables_bits: UInt64[jnp.ndarray, " n_op n_qubits"]
                The bits of the Pauli operators to apply.
            circuit_bit: UInt64[jnp.ndarray, "1 n_qubits"]
                The bit of the Pauli operator to apply.
            observables_signs: Complex128[jnp.ndarray, " n_op"]
                The signs of the Pauli operators to apply.
            circuit_sign: Complex128[jnp.ndarray, "1 n_qubits"]
                The sign of the Pauli operator to apply.
            observables_coefficients: Complex128[jnp.ndarray, " n_op"]
                The coefficients of the Pauli operators to apply.
            parameter: jnp.float64
                The parameter of the circuit.

        Returns:
            tuple[
                UInt64[jnp.ndarray, "n_op+n n_qubits"],
                Complex128[jnp.ndarray, " n_op+n"],
                Complex128[jnp.ndarray, " n_op+n"],
            ]:
                The new Pauli bits, the new Pauli signs, and the new Pauli coefficients.

        """
        anticommuting = jnp.where(
            anticommutes(
                self.observables_bits,
                circuit_bit,
                self.n_qubits,
            ),
        )[0]
        if len(anticommuting):
            new_bits, new_signs, pauli_indices, pauli_in_observables = self.multiply_operators(
                observables_bits=observables_bits,
                operator_bit=circuit_bit,
                observables_signs=observables_signs,
                operator_sign=circuit_sign,
                anticommuting_indices=anticommuting,
            )
            coeffs_sin: jnp.ndarray = observables_coefficients[anticommuting]
            coeffs_sin = pmult(coeffs_sin, (1j) * jnp.sin(2 * parameter))

            new_coeffs: jnp.ndarray = update_coeffs(
                observables_coefficients,
                observables_coefficients[pauli_indices % observables_bits.shape[0]],
                jnp.cos(2 * parameter),
                jnp.sin(2 * parameter),
                new_signs,
                observables_signs[pauli_indices % observables_bits.shape[0]],
                anticommuting,
                pauli_in_observables,
            )
            observables_coefficients = new_coeffs

            to_remove = a_lt_b(new_coeffs, self.threshold)
            if jnp.any(to_remove):
                observables_bits, observables_signs, observables_coefficients = delete_paulioperators(
                    bits=observables_bits,
                    signs=observables_signs,
                    coefficients=observables_coefficients,
                    index=anticommuting[to_remove],
                )

            to_add = a_gt_b_and_not_c(
                coeffs_sin,
                self.threshold,
                pauli_in_observables,
            )
            if jnp.any(to_add):
                observables_bits, observables_signs, observables_coefficients = self.add_new_paulis(
                    observables_bits=observables_bits,
                    new_bits=new_bits,
                    observables_signs=observables_signs,
                    new_signs=new_signs,
                    observables_coefficients=observables_coefficients,
                    new_coeffs=coeffs_sin,
                    ind_to_add=to_add,
                )

        return observables_bits, observables_signs, observables_coefficients

    def multiply_operators(
        self,
        observables_bits: UInt64[jnp.ndarray, "n_op n_qubits"],
        operator_bit: UInt64[jnp.ndarray, "1 n_qubits"],
        observables_signs: Complex128[jnp.ndarray, " n_op"],
        operator_sign: Complex128[jnp.ndarray, "1 n_qubits"],
        anticommuting_indices: jnp.ndarray,
    ) -> tuple[
        UInt64[jnp.ndarray, "n_op_new n_qubits"],
        Complex128[jnp.ndarray, " n_op_new"],
        UInt64[jnp.ndarray, " n_op_new"],
        Bool[jnp.ndarray, " n_op_new"],
    ]:
        """Multiply the operators in the anticommuting_indices with the operator_bit.

        Args:
            observables_bits: UInt64[jnp.ndarray, "n_op n_qubits"]
                The bits of the Pauli operators to multiply with.
            operator_bit: UInt64[jnp.ndarray, "1 n_qubits"]
                The bit of the operator to multiply with.
            observables_signs: Complex128[jnp.ndarray, " n_op"]
                The signs of the Pauli operators to multiply with.
            operator_sign: Complex128[jnp.ndarray, "1 n_qubits"]
                The sign of the operator to multiply with.
            anticommuting_indices: jnp.ndarray
                The indices of the operators to multiply with.

        Returns:
            tuple[
                UInt64[jnp.ndarray, "n_op_new n_qubits"],
                Complex128[jnp.ndarray, "n_op_new"],
                UInt64[jnp.ndarray, " n_op_new"],
                Bool[jnp.ndarray, " n_op_new"],
            ]:
                The new Pauli bits, the new Pauli signs, the indices of the new Pauli in the
                observables, and the boolean array indicating if the new Pauli is in the observables.

        """
        new_pauli_operators, new_pauli_signs = compose_with(
            observables_bits[anticommuting_indices, :],
            operator_bit,
            observables_signs[anticommuting_indices],
            operator_sign,
            self.n_qubits,
        )

        new_pauli_indices = find_paulioperators_indices(
            observables_bits,
            new_pauli_operators,
        )
        new_pauli_in_observables = find_paulioperators(
            observables_bits,
            new_pauli_operators,
            new_pauli_indices,
        )

        return new_pauli_operators, new_pauli_signs, new_pauli_indices, new_pauli_in_observables

    def add_new_paulis(
        self,
        observables_bits: UInt64[jnp.ndarray, "n_op n_qubits"],
        new_bits: UInt64[jnp.ndarray, "n_op n_qubits"],
        observables_signs: Complex128[jnp.ndarray, " n_op"],
        new_signs: Complex128[jnp.ndarray, " n_op"],
        observables_coefficients: Complex128[jnp.ndarray, " n_op"],
        new_coeffs: Complex128[jnp.ndarray, " n_op"],
        ind_to_add: UInt64[jnp.ndarray, " n"],
    ) -> tuple[
        UInt64[jnp.ndarray, "n_op+n n_qubits"],
        Complex128[jnp.ndarray, " n_op+n"],
        Complex128[jnp.ndarray, " n_op+n"],
    ]:
        """Add rows of new_paulis at indices ind_to_add to self.observable.

        These include Paulis that are above threshold and don't exist already in self.observable.

        Args:
            observables_bits: UInt64[jnp.ndarray, "n_op n_qubits"]
                The bits of the Pauli operators to add.
            new_bits: UInt64[jnp.ndarray, "n_op n_qubits"]
                The bits of the Pauli operators to add.
            observables_signs: Complex128[jnp.ndarray, " n_op"]
                The signs of the Pauli operators to add.
            new_signs: Complex128[jnp.ndarray, " n_op"]
                The signs of the Pauli operators to add.
            observables_coefficients: Complex128[jnp.ndarray, " n_op"]
                The coefficients of the Pauli operators to add.
            new_coeffs: Complex128[jnp.ndarray, " n_op"]
                The coefficients of the Pauli operators to add.
            ind_to_add: UInt64[jnp.ndarray, " n"]
                The indices of the Pauli operators to add.

        Returns:
            tuple[
                UInt64[jnp.ndarray, "n_op+n n_qubits"],
                Complex128[jnp.ndarray, " n_op+n"],
                Complex128[jnp.ndarray, " n_op+n"],
            ]:
                The new Pauli bits, the new Pauli signs, and the new Pauli coefficients.

        """
        new_bits, new_signs, new_coeffs = order_paulioperators(
            new_bits[ind_to_add, :],
            new_signs[ind_to_add],
            new_coeffs[ind_to_add],
        )

        # Insert new Paulis and return new array of coefficients.
        (
            observables_bits,
            observables_signs,
            observables_coefficients,
        ) = insert_paulioperators(
            bits=observables_bits,
            other_bits=new_bits,
            signs=observables_signs,
            other_signs=new_signs,
            coefficients=observables_coefficients,
            other_coefficients=new_coeffs,
            index=ind_to_add,
        )

        return observables_bits, observables_signs, observables_coefficients


@jax.jit
def pmult(
    a: jnp.ndarray,
    b: jnp.ndarray,
) -> jnp.ndarray:
    return a * b


@jax.jit
def update_coeffs(
    coeffs1: Complex128[jnp.ndarray, " n_op"],
    coeffs2: Complex128[jnp.ndarray, " n_op_new"],
    c: jnp.float64,
    s: jnp.float64,
    new_signs: Complex128[jnp.ndarray, " n_op_new"],
    signs: Complex128[jnp.ndarray, " n_op_new"],
    index_anticommuting: UInt64[jnp.ndarray, " n_anticommuting"],
    index_exists: Bool[jnp.ndarray, " n_op_new"],
) -> Complex128[jnp.ndarray, " n_op"]:
    tmp = coeffs2 * (index_exists * (1j) * s * new_signs / signs)
    return coeffs1.at[index_anticommuting].set(coeffs1.at[index_anticommuting].get() * c + tmp)


@jax.jit
def a_lt_b(a: jnp.ndarray, b: float) -> jnp.ndarray:
    """Compare absolute values of vector elements with a scalar.

    Args:
        a: Vector input array
        b: Scalar threshold value

    Returns:
        Boolean array where True indicates |a[i]| < b

    """
    return jnp.abs(a) < b


@jax.jit
def a_gt_b_and_not_c(a: jnp.ndarray, b: float, c: jnp.ndarray) -> jnp.ndarray:
    return (jnp.abs(a) >= b) & ~c
