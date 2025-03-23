import copy
from collections.abc import Callable
from typing import Self

import jax.numpy as jnp
from jaxtyping import Bool, Complex128, Float64, UInt64

from paulistringsquantumcircuitsimulations.circuit import Circuit
from paulistringsquantumcircuitsimulations.exceptions import InvalidParameterError, SystemSizeError
from paulistringsquantumcircuitsimulations.exported import (
    anticommutation_exported,
    bits_equal_exported,
    count_nonzero_exported,
    find_bit_index_exported,
    new_sign_exported,
    not_equal_exported,
    pack_bits_exported,
    xor_exported,
    ztype_bool_exported,
)
from paulistringsquantumcircuitsimulations.paulioperators import (
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
from paulistringsquantumcircuitsimulations.utils import PauliString


class HeisenbergSimulator:
    """A class for simulating Heisenberg models.

    Args:
        n_qubits (int): The number of qubits in the circuit.
        operator_bit_list (list[UInt64[jnp.ndarray, "1 n_packed"]]): The bits of the operators.
        operator_sign_list (list[Complex128[jnp.ndarray, "1 n_packed"]]): The signs of the operators.
        observables_bits (UInt64[jnp.ndarray, " n_op n_packed"]): The bits of the observables.
        observables_signs (Complex128[jnp.ndarray, " n_op"]): The signs of the observables.
        observables_coefficients (Complex128[jnp.ndarray, " n_op"]): The coefficients of the observables.
        operator_real_coefficients (Float64[jnp.ndarray, " n_circuit_parameters"]):
            The real coefficients of the operators.
        threshold (float): The threshold for the Pauli strings.

    """

    def __init__(
        self,
        n_qubits: int,
        operator_bit_list: list[UInt64[jnp.ndarray, "1 2n_packed"]],
        operator_sign_list: list[Complex128[jnp.ndarray, " 1"]],
        observables_bits: UInt64[jnp.ndarray, " n_op 2n_packed"],
        observables_signs: Complex128[jnp.ndarray, " n_op"],
        observables_coefficients: Complex128[jnp.ndarray, " n_op"],
        operator_real_coefficients: Float64[jnp.ndarray, " n_circuit_parameters"],
        _exports: dict[str, Callable],
        threshold: float = 0.0,
    ) -> None:
        """Initialize the HeisenbergSimulator.

        Args:
            n_qubits (int): The number of qubits in the circuit.
            operator_bit_list (list[UInt64[jnp.ndarray, "1 2n_packed"]]): The bits of the operators.
            operator_sign_list (list[Complex128[jnp.ndarray, "1 2n_packed"]]): The signs of the operators.
            observables_bits (UInt64[jnp.ndarray, " n_op 2n_packed"]): The bits of the observables.
            observables_signs (Complex128[jnp.ndarray, " n_op"]): The signs of the observables.
            observables_coefficients (Complex128[jnp.ndarray, " n_op"]): The coefficients of the observables.
            operator_real_coefficients (Float64[jnp.ndarray, " n_circuit_parameters"]):
                The real coefficients of the operators.
                Note that `n_circuit_parameters` corresponds to the length of `operator_bit_list` and
                `operator_sign_list`.
            _exports (dict[str, Callable]): The exported methods.
            threshold (float): The threshold for the Pauli strings.

        """
        self.n_qubits = n_qubits
        self.operator_bit_list = operator_bit_list
        self.operator_sign_list = operator_sign_list
        self.observables_bits = observables_bits
        self.observables_signs = observables_signs
        self.observables_coefficients = observables_coefficients
        self.operator_real_coefficients = operator_real_coefficients
        self.exports = _exports
        self.threshold = threshold

        self.exports["xor"] = xor_exported(n_qubits)
        self.exports["sign"] = new_sign_exported(n_qubits)
        self.exports["find_bit_index"] = find_bit_index_exported(n_qubits)
        self.exports["bits_equal"] = bits_equal_exported(n_qubits)
        self.exports["anticommutation"] = anticommutation_exported(n_qubits)
        self.exports["not_equal"] = not_equal_exported()
        self.exports["ztype_bool"] = ztype_bool_exported(n_qubits)
        self.exports["count_nonzero"] = count_nonzero_exported(n_qubits)
        self.exports["expose_ztype_bool"] = ztype_bool_exported(n_qubits)

    @classmethod
    def init_circuit(
        cls,
        n_qubits: int,
        circuit: Circuit,
        paulistrings: list[PauliString],
        coefficients: Complex128[jnp.ndarray, " n_op"]
        | Float64[jnp.ndarray, " n_op"]
        | list[complex]
        | list[float]
        | None = None,
        threshold: float = 0.0,
    ) -> Self:
        """Initialize the HeisenbergSimulator from a circuit.

        Args:
            n_qubits (int): The number of qubits in the circuit.
            circuit (Circuit): The circuit to simulate.
            paulistrings (list[PauliString]): The Pauli strings to simulate.
            coefficients: The coefficients of the Pauli strings. Can be:
                - Complex128[jnp.ndarray, " n_op"]: Complex coefficients as JAX array
                - Float64[jnp.ndarray, " n_op"]: Real coefficients as JAX array
                - list[complex]: Complex coefficients as Python list
                - list[float]: Real coefficients as Python list
                - None: Default to all ones
            threshold (float): The threshold for the Pauli strings.

        Returns:
            HeisenbergSimulator: The Heisenberg simulator.

        """
        if n_qubits != circuit.n_qubits:
            raise SystemSizeError(n_qubits, circuit.n_qubits)

        _exports = {
            "pack_bits": pack_bits_exported(n_qubits),
        }

        circuit_paulistrings, circuit_signs = circuit.get_paulistrings()

        operator_bit_list: list[UInt64[jnp.ndarray, "1 2n_packed"]] = []
        operator_sign_list: list[Complex128[jnp.ndarray, " 1"]] = []
        for i in range(len(circuit_paulistrings)):
            circuit_bit, circuit_sign, _ = paulioperators_from_strings(
                n_qubits=n_qubits,
                paulistrings=[circuit_paulistrings[i]],
                signs=jnp.array([circuit_signs[i]], dtype=jnp.complex128),
                pack_bits=_exports["pack_bits"],
            )
            operator_bit_list.append(circuit_bit)
            operator_sign_list.append(circuit_sign)

        operator_real_coefficients = jnp.ones(len(circuit_paulistrings), dtype=jnp.float64)
        paulistrings, signs = circuit.transform_paulistrings(
            paulistrings=paulistrings,
        )

        observables_bits, observables_signs, observables_coefficients = paulioperators_from_strings(
            n_qubits=n_qubits,
            paulistrings=paulistrings,
            signs=signs,
            coefficients=coefficients,
            pack_bits=_exports["pack_bits"],
        )
        observables_bits, observables_signs, observables_coefficients = order_paulioperators(
            bits=observables_bits,
            signs=observables_signs,
            coefficients=observables_coefficients,
        )
        return cls(
            n_qubits=n_qubits,
            operator_bit_list=operator_bit_list,
            operator_sign_list=operator_sign_list,
            observables_bits=observables_bits,
            observables_signs=observables_signs,
            observables_coefficients=observables_coefficients,
            operator_real_coefficients=operator_real_coefficients,
            _exports=_exports,
            threshold=threshold,
        )

    @classmethod
    def init_real_dynamics(
        cls,
        n_qubits: int,
        operator_paulistrings: list[PauliString],
        operator_real_coefficients: Float64[jnp.ndarray, " n_circuit_parameters"],
        paulistrings: list[PauliString],
        coefficients: Complex128[jnp.ndarray, " n_op"]
        | Float64[jnp.ndarray, " n_op"]
        | list[complex]
        | list[float]
        | None = None,
        threshold: float = 0.0,
    ) -> Self:
        """Initialize the HeisenbergSimulator from a real Hamiltonian.

        Args:
            n_qubits (int): The number of qubits in the circuit.
            operator_paulistrings (list[PauliString]): The Pauli strings of the operators.
            operator_real_coefficients (Float64[jnp.ndarray, " n_circuit_parameters"]):
                The real coefficients of the operators.
                Note that `n_circuit_parameters` corresponds to the length of `operator_paulistrings`.
            paulistrings (list[PauliString]): The Pauli strings of the observables.
            coefficients: The coefficients of the observables. Can be:
                - Complex128[jnp.ndarray, " n_op"]: Complex coefficients as JAX array
                - Float64[jnp.ndarray, " n_op"]: Real coefficients as JAX array
                - list[complex]: Complex coefficients as Python list
                - list[float]: Real coefficients as Python list
                - None: Default to all ones
            threshold (float): The threshold for the Pauli strings.

        Returns:
            HeisenbergSimulator: The Heisenberg simulator.

        """
        _exports = {
            "pack_bits": pack_bits_exported(n_qubits),
        }

        operator_bit_list: list[UInt64[jnp.ndarray, "1 2n_packed"]] = []
        operator_sign_list: list[Complex128[jnp.ndarray, " 1"]] = []
        for i in range(len(operator_paulistrings)):
            circuit_bit, circuit_sign, _ = paulioperators_from_strings(
                n_qubits=n_qubits,
                paulistrings=[operator_paulistrings[i]],
                signs=jnp.array([operator_real_coefficients[i]], dtype=jnp.complex128),
                pack_bits=_exports["pack_bits"],
            )
            operator_bit_list.append(circuit_bit)
            operator_sign_list.append(circuit_sign)

        observables_bits, observables_signs, observables_coefficients = paulioperators_from_strings(
            n_qubits=n_qubits,
            paulistrings=paulistrings,
            coefficients=coefficients,
            pack_bits=_exports["pack_bits"],
        )

        observables_bits, observables_signs, observables_coefficients = order_paulioperators(
            bits=observables_bits,
            signs=observables_signs,
            coefficients=observables_coefficients,
        )

        return cls(
            n_qubits=n_qubits,
            operator_bit_list=operator_bit_list,
            operator_sign_list=operator_sign_list,
            observables_bits=observables_bits,
            observables_signs=observables_signs,
            observables_coefficients=observables_coefficients,
            operator_real_coefficients=jnp.array(operator_real_coefficients, dtype=jnp.float64),
            _exports=_exports,
            threshold=threshold,
        )

    def run(
        self,
        observables_bits: UInt64[jnp.ndarray, " n_op 2n_packed"],
        observables_signs: Complex128[jnp.ndarray, " n_op"],
        observables_coefficients: Complex128[jnp.ndarray, " n_op"],
        parameters: Float64[jnp.ndarray, " n_circuit_parameters"],
    ) -> tuple[
        UInt64[jnp.ndarray, " n_op_new 2n_packed"],
        Complex128[jnp.ndarray, " n_op_new"],
        Complex128[jnp.ndarray, " n_op_new"],
    ]:
        """Run the Heisenberg simulator to update the observables.

        Args:
            observables_bits (UInt64[jnp.ndarray, " n_op 2n_packed"]):
                The bits of the observables.
            observables_signs (Complex128[jnp.ndarray, " n_op"]):
                The signs of the observables.
            observables_coefficients (Complex128[jnp.ndarray, " n_op"]):
                The coefficients of the observables.
            parameters (Float64[jnp.ndarray, " n_circuit_parameters"]):
                The parameters of the circuit.

        Returns:
            tuple[
                UInt64[jnp.ndarray, " n_op_new 2n_packed"],
                Complex128[jnp.ndarray, " n_op_new"],
                Complex128[jnp.ndarray, " n_op_new"],
            ]:
                The Pauli bits, the Pauli signs, and the Pauli coefficients.
                These lengths are updated by this function.

        """
        if parameters.shape[0] != len(self.operator_bit_list):
            raise InvalidParameterError(len(self.operator_bit_list), parameters.shape[0])

        for i in range(parameters.shape[0]):
            observables_bits, observables_signs, observables_coefficients = self.apply_pauli_operator(
                observables_bits=observables_bits,
                operator_bit=self.operator_bit_list[i],
                observables_signs=observables_signs,
                operator_sign=self.operator_sign_list[i],
                observables_coefficients=observables_coefficients,
                parameter=parameters[i],
            )
        return observables_bits, observables_signs, observables_coefficients

    def run_circuit(
        self,
        parameters: Float64[jnp.ndarray, " n_circuit_parameters"],
    ) -> jnp.float64:
        """Run the Heisenberg simulator on a circuit and compute the expectation value.

        Args:
            parameters (Float64[jnp.ndarray, " n_circuit_parameters"]):
                The parameters of the circuit.

        Returns:
            jnp.float64: The expectation value of the observables.

        """
        observables_bits, observables_signs, observables_coefficients = self.run(
            observables_bits=copy.deepcopy(self.observables_bits),
            observables_signs=copy.deepcopy(self.observables_signs),
            observables_coefficients=copy.deepcopy(self.observables_coefficients),
            parameters=parameters,
        )

        return evaluate_expectation_value_zero_state(
            heisenberg_simulator=self,
            observables_bits=observables_bits,
            observables_signs=observables_signs,
            observables_coefficients=observables_coefficients,
        )

    def run_dynamics(
        self,
        nsteps: int,
        dt: float,
        process: Callable | None = None,
        process_every: int = 1,
    ) -> Float64[jnp.ndarray, " nprocs"]:
        r"""Run the dynamics based on the Heisenberg simulator and evaluate the process function.

        Args:
            nsteps (int): The number of steps to run the dynamics.
            dt (jnp.float64): The time step.
            process (Callable | None): The process function to evaluate.
                It takes the observables bits, the observables signs, and the observables coefficients.
                Note that if `process` is `None`, the process function is the expectation value of the
                observables with $\ket{0}^{\otimes n}$ as the initial state.
            process_every (int): The number of steps to evaluate the process function.

        Returns:
            Float64[jnp.ndarray, " nprocs"]: The process function evaluated at each step.

        """
        r = []

        observables_bits, observables_signs, observables_coefficients = (
            copy.deepcopy(self.observables_bits),
            copy.deepcopy(self.observables_signs),
            copy.deepcopy(self.observables_coefficients),
        )
        if process is None:
            process = evaluate_expectation_value_zero_state

        r.append(
            process(
                heisenberg_simulator=self,
                observables_bits=observables_bits,
                observables_signs=observables_signs,
                observables_coefficients=observables_coefficients,
            )
        )
        for step in range(nsteps):
            observables_bits, observables_signs, observables_coefficients = self.run(
                observables_bits=observables_bits,
                observables_signs=observables_signs,
                observables_coefficients=observables_coefficients,
                parameters=jnp.float64(dt) * self.operator_real_coefficients,
            )
            if (step + 1) % process_every == 0:
                r.append(
                    process(
                        heisenberg_simulator=self,
                        observables_bits=observables_bits,
                        observables_signs=observables_signs,
                        observables_coefficients=observables_coefficients,
                    )
                )
        return jnp.array(r, dtype=jnp.float64)

    def apply_pauli_operator(
        self,
        observables_bits: UInt64[jnp.ndarray, " n_op 2n_packed"],
        operator_bit: UInt64[jnp.ndarray, "1 2n_packed"],
        observables_signs: Complex128[jnp.ndarray, " n_op"],
        operator_sign: Complex128[jnp.ndarray, "1 2n_packed"],
        observables_coefficients: Complex128[jnp.ndarray, " n_op"],
        parameter: jnp.float64,
    ) -> tuple[
        UInt64[jnp.ndarray, "n_op+n 2n_packed"],
        Complex128[jnp.ndarray, " n_op+n"],
        Complex128[jnp.ndarray, " n_op+n"],
    ]:
        """Apply a Pauli operator to the circuit.

        Args:
            observables_bits (UInt64[jnp.ndarray, " n_op 2n_packed"]):
                The bits of the Pauli operators to apply.
            operator_bit (UInt64[jnp.ndarray, "1 2n_packed"]):
                The bit of the Pauli operator to apply.
            observables_signs (Complex128[jnp.ndarray, " n_op"]):
                The signs of the Pauli operators to apply.
            operator_sign (Complex128[jnp.ndarray, "1 2n_packed"]):
                The sign of the Pauli operator to apply.
            observables_coefficients (Complex128[jnp.ndarray, " n_op"]):
                The coefficients of the Pauli operators to apply.
            parameter (jnp.float64):
                The parameter of the circuit.

        Returns:
            tuple[
                UInt64[jnp.ndarray, "n_op+n 2n_packed"],
                Complex128[jnp.ndarray, " n_op+n"],
                Complex128[jnp.ndarray, " n_op+n"],
            ]:
                The new Pauli bits, the new Pauli signs, and the new Pauli coefficients.

        """
        anticommuting = jnp.where(
            anticommutes(
                n_qubits=self.n_qubits,
                bits=observables_bits,
                other_bit=operator_bit,
                anticommutation=self.exports["anticommutation"],
                not_equal=self.exports["not_equal"],
            ),
        )[0]
        if len(anticommuting):
            new_bits, new_signs, pauli_indices, pauli_in_observables = self.multiply_operators(
                observables_bits=observables_bits,
                operator_bit=operator_bit,
                observables_signs=observables_signs,
                operator_sign=operator_sign,
                anticommuting_indices=anticommuting,
            )
            coeffs_sin: jnp.ndarray = observables_coefficients[anticommuting]
            coeffs_sin = (1j) * jnp.sin(2 * parameter) * coeffs_sin  # TODO: check it can be jitted

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
                    observables_signs=observables_signs,
                    observables_coefficients=observables_coefficients,
                    new_bits=new_bits,
                    new_signs=new_signs,
                    new_coeffs=coeffs_sin,
                    ind_to_add=to_add,
                )

        return observables_bits, observables_signs, observables_coefficients

    def multiply_operators(
        self,
        observables_bits: UInt64[jnp.ndarray, "n_op 2n_packed"],
        operator_bit: UInt64[jnp.ndarray, "1 2n_packed"],
        observables_signs: Complex128[jnp.ndarray, " n_op"],
        operator_sign: Complex128[jnp.ndarray, "1 2n_packed"],
        anticommuting_indices: jnp.ndarray,
    ) -> tuple[
        UInt64[jnp.ndarray, "n_op_new 2n_packed"],
        Complex128[jnp.ndarray, " n_op_new"],
        UInt64[jnp.ndarray, " n_op_new"],
        Bool[jnp.ndarray, " n_op_new"],
    ]:
        """Multiply the operators in the anticommuting_indices with the operator_bit.

        Args:
            observables_bits (UInt64[jnp.ndarray, "n_op 2n_packed"]):
                The bits of the Pauli operators to multiply with.
            operator_bit (UInt64[jnp.ndarray, "1 2n_packed"]):
                The bit of the operator to multiply with.
            observables_signs (Complex128[jnp.ndarray, " n_op"]):
                The signs of the Pauli operators to multiply with.
            operator_sign (Complex128[jnp.ndarray, "1 2n_packed"]):
                The sign of the operator to multiply with.
            anticommuting_indices (jnp.ndarray):
                The indices of the operators to multiply with.

        Returns:
            tuple[
                UInt64[jnp.ndarray, "n_op_new 2n_packed"],
                Complex128[jnp.ndarray, "n_op_new"],
                UInt64[jnp.ndarray, " n_op_new"],
                Bool[jnp.ndarray, " n_op_new"],
            ]:
                The new Pauli bits, the new Pauli signs, the indices of the new Pauli in the
                observables, and the boolean array indicating if the new Pauli is in the observables.

        """
        new_pauli_operators, new_pauli_signs = compose_with(
            self.n_qubits,
            observables_bits[anticommuting_indices, :],
            operator_bit,
            observables_signs[anticommuting_indices],
            operator_sign,
            xor=self.exports["xor"],
            new_sign=self.exports["sign"],
        )

        new_pauli_indices = find_paulioperators_indices(
            observables_bits,
            new_pauli_operators,
            find_bit_index=self.exports["find_bit_index"],
        )
        new_pauli_in_observables = find_paulioperators(
            observables_bits,
            new_pauli_operators,
            new_pauli_indices,
            bits_equal=self.exports["bits_equal"],
        )

        return new_pauli_operators, new_pauli_signs, new_pauli_indices, new_pauli_in_observables

    def add_new_paulis(
        self,
        observables_bits: UInt64[jnp.ndarray, "n_op 2n_packed"],
        observables_signs: Complex128[jnp.ndarray, " n_op"],
        observables_coefficients: Complex128[jnp.ndarray, " n_op"],
        new_bits: UInt64[jnp.ndarray, "n_op 2n_packed"],
        new_signs: Complex128[jnp.ndarray, " n_op"],
        new_coeffs: Complex128[jnp.ndarray, " n_op"],
        ind_to_add: UInt64[jnp.ndarray, " n"],
    ) -> tuple[
        UInt64[jnp.ndarray, "n_op+n 2n_packed"],
        Complex128[jnp.ndarray, " n_op+n"],
        Complex128[jnp.ndarray, " n_op+n"],
    ]:
        """Add rows of new_paulis at indices ind_to_add to self.observable.

        These include Paulis that are above threshold and don't exist already in self.observable.

        Args:
            observables_bits (UInt64[jnp.ndarray, "n_op 2n_packed"]):
                The bits of the Pauli operators to add.
            new_bits (UInt64[jnp.ndarray, "n_op 2n_packed"]):
                The bits of the Pauli operators to add.
            observables_signs (Complex128[jnp.ndarray, " n_op"]):
                The signs of the Pauli operators to add.
            new_signs (Complex128[jnp.ndarray, " n_op"]):
                The signs of the Pauli operators to add.
            observables_coefficients (Complex128[jnp.ndarray, " n_op"]):
                The coefficients of the Pauli operators to add.
            new_coeffs (Complex128[jnp.ndarray, " n_op"]):
                The coefficients of the Pauli operators to add.
            ind_to_add (UInt64[jnp.ndarray, " n"]):
                The indices of the Pauli operators to add.

        Returns:
            tuple[
                UInt64[jnp.ndarray, "n_op+n 2n_packed"],
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
            find_bit_index=self.exports["find_bit_index"],
        )

        return observables_bits, observables_signs, observables_coefficients


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
    tmp = coeffs2 * (index_exists * (1j) * s * signs / new_signs)
    return coeffs1.at[index_anticommuting].set(coeffs1.at[index_anticommuting].get() * c + tmp)


def a_lt_b(a: Complex128[jnp.ndarray, " n_op"], b: jnp.float64) -> Bool[jnp.ndarray, " n_op"]:
    result: Bool[jnp.ndarray, " n_op"] = jnp.abs(a) < b
    return result


def a_gt_b_and_not_c(
    a: Complex128[jnp.ndarray, " n_op"], b: jnp.float64, c: Bool[jnp.ndarray, " n_op"]
) -> Bool[jnp.ndarray, " n_op"]:
    result: Bool[jnp.ndarray, " n_op"] = (jnp.abs(a) >= b) & ~c
    return result


def evaluate_expectation_value_zero_state(
    heisenberg_simulator: HeisenbergSimulator,
    observables_bits: UInt64[jnp.ndarray, " n_op 2n_packed"],
    observables_signs: Complex128[jnp.ndarray, " n_op"],
    observables_coefficients: Complex128[jnp.ndarray, " n_op"],
) -> Complex128[jnp.ndarray, " n"]:
    """Evaluate Pauli expectation values with respect to the |0> state.

    Args:
        heisenberg_simulator: HeisenbergSimulator
            The Heisenberg simulator to evaluate the expectation value of.
        observables_bits (UInt64[jnp.ndarray, " n_op 2n_packed"]):
            The bits of the Pauli operators to evaluate the expectation value of.
        observables_signs (Complex128[jnp.ndarray, " n_op"]):
            The signs of the Pauli operators to evaluate the expectation value of.
        observables_coefficients (Complex128[jnp.ndarray, " n_op"]):
            The coefficients of the Pauli operators to evaluate the expectation value of.

    Returns:
        Array[" n", jnp.complex128]: The expectation value of the Pauli operators.

    """
    nonzero_pauli_indices = jnp.where(
        ztype(
            heisenberg_simulator.n_qubits,
            observables_bits,
            ztype_bool=heisenberg_simulator.exports["ztype_bool"],
        )
    )[0]
    return jnp.real(
        jnp.sum(observables_coefficients[nonzero_pauli_indices] * observables_signs[nonzero_pauli_indices]),
    )


def overlap(
    heisenberg_simulator: HeisenbergSimulator,
    observables_bits: UInt64[jnp.ndarray, " n_op 2n_packed"],
    observables_signs: Complex128[jnp.ndarray, " n_op"],
    observables_coefficients: Complex128[jnp.ndarray, " n_op"],
    other_bits: UInt64[jnp.ndarray, "n_op_others 2n_packed"],
    other_signs: Complex128[jnp.ndarray, " n_op_others"],
    other_coefficients: Complex128[jnp.ndarray, " n_op_others"],
) -> Complex128[jnp.ndarray, " n_op_others"]:
    """Compute overlap of two Pauli sums as Tr[B^dag A] / N, where N is a normalization factor.

    Bits (A) and other (B) are both PauliRepresentation objects.

    Args:
        heisenberg_simulator: HeisenbergSimulator
            The Heisenberg simulator to compute the overlap of.
        observables_bits (UInt64[jnp.ndarray, "n_op 2n_packed"]):
            The bits of the Pauli operators.
        observables_signs (Complex128[jnp.ndarray, " n_op"]):
            The signs of the Pauli operators.
        observables_coefficients (Complex128[jnp.ndarray, " n_op"]):
            The coefficients of the Pauli operators.
        other_bits (UInt64[jnp.ndarray, "n_op_others 2n_packed"]):
            The bits of the Pauli operators to compute the overlap with.
        other_signs (Complex128[jnp.ndarray, " n_op_others"]):
            The signs of the Pauli operators to compute the overlap with.
        other_coefficients (Complex128[jnp.ndarray, " n_op_others"]):
            The coefficients of the Pauli operators to compute the overlap with.

    Returns:
        Complex128[jnp.ndarray, " n_op_others"]:
            The overlap of the Pauli operators.

    """
    index = find_paulioperators_indices(
        observables_bits,
        other_bits,
        find_bit_index=heisenberg_simulator.exports["find_bit_index"],
    )
    pauli_found = find_paulioperators(
        observables_bits,
        other_bits,
        index=index,
        bits_equal=heisenberg_simulator.exports["bits_equal"],
    )
    index_found = index[pauli_found]
    return jnp.sum(
        observables_coefficients[index_found]
        * jnp.conj(other_coefficients[pauli_found])
        * observables_signs[index_found]
        / other_signs[pauli_found]
    )
