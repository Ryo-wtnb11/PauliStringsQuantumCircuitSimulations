import copy
from collections.abc import Callable
from typing import Self

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Complex64, Float32, Int32, UInt32
from numba import set_num_threads

from paulistringsquantumcircuitsimulations.circuit import Circuit
from paulistringsquantumcircuitsimulations.exceptions import InvalidParameterError, SystemSizeError
from paulistringsquantumcircuitsimulations.paulioperators import (
    PauliString,
    anticommutes,
    coefs_gt_threshold_and_not_logic,
    coefs_lt_threshold,
    compose_with,
    delete_paulioperators,
    find_paulioperators,
    find_paulioperators_indices,
    insert_paulioperators,
    order_paulioperators,
    paulioperators_from_strings,
    update_coeffs,
    ztype,
)

set_num_threads(3)


class HeisenbergSimulator:
    def __init__(
        self,
        n_qubits: int,
        operator_bit_list: list[UInt32[np.ndarray, "1 n_packed"]],
        operator_phase_list: list[Int32[np.ndarray, " 1"]],
        observables_bits: UInt32[np.ndarray, "n_operators n_packed"],
        observables_phases: Int32[np.ndarray, " n_operators"],
        observables_coefficients: Complex64[jnp.ndarray, " n_operators"],
        operator_real_coefficients: Float32[jnp.ndarray, " n_circuit_parameters"],
        threshold: float = 0.0,
    ) -> None:
        self.n_qubits = n_qubits
        self.operator_bit_list = operator_bit_list
        self.operator_phase_list = operator_phase_list
        self.observables_bits = observables_bits
        self.observables_phases = observables_phases
        self.observables_coefficients = observables_coefficients
        self.operator_real_coefficients = operator_real_coefficients
        self.threshold = threshold

    @classmethod
    def init_circuit(
        cls,
        n_qubits: int,
        circuit: Circuit,
        paulistrings: list[PauliString],
        coefficients: list[complex] | list[float] | None = None,
        threshold: float = 0.0,
    ) -> Self:
        if n_qubits != circuit.n_qubits:
            raise SystemSizeError(n_qubits, circuit.n_qubits)

        circuit_paulistrings, circuit_phases = circuit.get_paulistrings()

        operator_bit_list: list[UInt32[np.ndarray, "1 n_packed"]] = []
        operator_phase_list: list[Int32[np.ndarray, " 1"]] = []
        for i in range(len(circuit_paulistrings)):
            circuit_bit, circuit_phase, _ = paulioperators_from_strings(
                n_qubits=n_qubits,
                paulistrings=[circuit_paulistrings[i]],
                phases=[circuit_phases[i]],
            )
            operator_bit_list.append(circuit_bit)
            operator_phase_list.append(circuit_phase)

        operator_real_coefficients = np.ones(len(circuit_paulistrings), dtype=np.float32)
        paulistrings, phases = circuit.transform_paulistrings(
            paulistrings=paulistrings,
        )

        observables_bits, observables_phases, observables_coefficients = paulioperators_from_strings(
            n_qubits=n_qubits,
            paulistrings=paulistrings,
            phases=phases,
            coefficients=coefficients,
        )
        observables_bits, observables_phases, observables_coefficients = order_paulioperators(
            bits=observables_bits,
            phases=observables_phases,
            coefficients=observables_coefficients,
        )
        return cls(
            n_qubits=n_qubits,
            operator_bit_list=operator_bit_list,
            operator_phase_list=operator_phase_list,
            observables_bits=observables_bits,
            observables_phases=observables_phases,
            observables_coefficients=observables_coefficients,
            operator_real_coefficients=jnp.array(operator_real_coefficients, dtype=jnp.float32),
            threshold=threshold,
        )

    @classmethod
    def init_real_dynamics(
        cls,
        n_qubits: int,
        operator_paulistrings: list[PauliString],
        operator_real_coefficients: list[float],
        paulistrings: list[PauliString],
        coefficients: list[complex] | list[float] | None = None,
        threshold: float = 0.0,
    ) -> Self:
        operator_bit_list: list[UInt32[np.ndarray, "1 n_packed"]] = []
        operator_phase_list: list[Int32[np.ndarray, " 1"]] = []
        for i in range(len(operator_paulistrings)):
            circuit_bit, circuit_phase, _ = paulioperators_from_strings(
                n_qubits=n_qubits,
                paulistrings=[operator_paulistrings[i]],
            )
            operator_bit_list.append(circuit_bit)
            operator_phase_list.append(circuit_phase)

        observables_bits, observables_phases, observables_coefficients = paulioperators_from_strings(
            n_qubits=n_qubits,
            paulistrings=paulistrings,
            coefficients=coefficients,
        )

        observables_bits, observables_phases, observables_coefficients = order_paulioperators(
            bits=observables_bits,
            phases=observables_phases,
            coefficients=observables_coefficients,
        )

        return cls(
            n_qubits=n_qubits,
            operator_bit_list=operator_bit_list,
            operator_phase_list=operator_phase_list,
            observables_bits=observables_bits,
            observables_phases=observables_phases,
            observables_coefficients=observables_coefficients,
            operator_real_coefficients=jnp.array(operator_real_coefficients, dtype=jnp.float32),
            threshold=threshold,
        )

    def run(
        self,
        observables_bits: UInt32[np.ndarray, "n_operators n_packed"],
        observables_phases: Int32[np.ndarray, " n_operators"],
        observables_coefficients: Complex64[jnp.ndarray, " n_operators"],
        parameters: Float32[jnp.ndarray, " n_circuit_parameters"],
    ) -> tuple[
        UInt32[np.ndarray, "n_operators n_packed"],
        Int32[np.ndarray, " n_operators"],
        Complex64[jnp.ndarray, " n_operators"],
    ]:
        if parameters.shape[0] != len(self.operator_bit_list):
            raise InvalidParameterError(len(self.operator_bit_list), parameters.shape[0])

        for i in range(parameters.shape[0]):
            observables_bits, observables_phases, observables_coefficients = self.apply_pauli_operator(
                observables_bits,
                observables_phases,
                observables_coefficients,
                self.operator_bit_list[i],
                self.operator_phase_list[i],
                parameter=parameters[i],
            )
        return observables_bits, observables_phases, observables_coefficients

    def run_circuit(
        self,
        parameters: Float32[Array, " n_circuit_parameters"],
    ) -> Float32[Array, ""]:
        observables_bits, observables_phases, observables_coefficients = self.run(
            observables_bits=copy.deepcopy(self.observables_bits),
            observables_phases=copy.deepcopy(self.observables_phases),
            observables_coefficients=copy.deepcopy(self.observables_coefficients),
            parameters=parameters,
        )

        return evaluate_expectation_value_zero_state(
            observables_bits=observables_bits,
            observables_phases=observables_phases,
            observables_coefficients=observables_coefficients,
        )

    def run_dynamics(
        self,
        nsteps: int,
        dt: float,
        process: Callable | None = None,
        process_every: int = 1,
    ) -> Float32[jnp.ndarray, " nsteps"]:
        r = []

        observables_bits, observables_phases, observables_coefficients = (
            copy.deepcopy(self.observables_bits),
            copy.deepcopy(self.observables_phases),
            copy.deepcopy(self.observables_coefficients),
        )
        if process is None:
            process = evaluate_expectation_value_zero_state
        r.append(
            process(
                observables_bits,
                observables_phases,
                observables_coefficients,
            )
        )
        for step in range(nsteps):
            observables_bits, observables_phases, observables_coefficients = self.run(
                observables_bits=observables_bits,
                observables_phases=observables_phases,
                observables_coefficients=observables_coefficients,
                parameters=dt * self.operator_real_coefficients,
            )
            if (step + 1) % process_every == 0:
                r.append(
                    process(
                        observables_bits,
                        observables_phases,
                        observables_coefficients,
                    )
                )
        return jnp.array(r, dtype=np.float32)

    def apply_pauli_operator(
        self,
        observables_bits: UInt32[np.ndarray, "n_operators n_packed"],
        observables_phases: Int32[np.ndarray, " n_operators"],
        observables_coefficients: Complex64[jnp.ndarray, " n_operators"],
        operator_bit: UInt32[np.ndarray, "n_operators n_packed"],
        operator_phase: Int32[np.ndarray, " n_operators"],
        parameter: jnp.float32,
    ) -> tuple[
        UInt32[np.ndarray, " n_operators n_packed"],
        Int32[np.ndarray, " n_operators"],
        Complex64[jnp.ndarray, " n_operators"],
    ]:
        anticommuting = np.where(
            anticommutes(
                observables_bits,
                operator_bit,
            ),
        )[0]
        if len(anticommuting):
            new_bits, new_phases, pauli_indices, pauli_in_observables = self.multiply_operators(
                observables_bits=observables_bits,
                operator_bit=operator_bit,
                observables_phases=observables_phases,
                operator_phase=operator_phase,
                anticommuting_indices=anticommuting,
            )
            coeffs_sin: Complex64[jnp.ndarray, " n_operators"] = observables_coefficients[anticommuting]
            coeffs_sin = (1j) * jnp.sin(2 * parameter) * coeffs_sin
            new_coeffs: Complex64[jnp.ndarray, " n_operators"] = update_coeffs(
                observables_coefficients,
                observables_coefficients[pauli_indices % observables_bits.shape[0]],
                jnp.cos(2 * parameter),
                jnp.sin(2 * parameter),
                new_phases,
                observables_phases[pauli_indices % observables_bits.shape[0]],
                anticommuting,
                pauli_in_observables,
            )

            observables_coefficients = new_coeffs
            to_remove = coefs_lt_threshold(observables_coefficients[anticommuting], self.threshold)
            if np.any(to_remove):
                observables_bits, observables_phases, observables_coefficients = delete_paulioperators(
                    bits=observables_bits,
                    phases=observables_phases,
                    coefficients=observables_coefficients,
                    index=anticommuting[to_remove],
                )

            to_add = coefs_gt_threshold_and_not_logic(
                coeffs_sin,
                self.threshold,
                pauli_in_observables,
            )
            if np.any(to_add):
                observables_bits, observables_phases, observables_coefficients = self.add_new_paulis(
                    observables_bits=observables_bits,
                    observables_phases=observables_phases,
                    observables_coefficients=observables_coefficients,
                    new_bits=new_bits,
                    new_phases=new_phases,
                    new_coeffs=coeffs_sin,
                    ind_to_add=to_add,
                )

        return observables_bits, observables_phases, observables_coefficients

    def multiply_operators(
        self,
        observables_bits: UInt32[np.ndarray, " n_operators n_packed"],
        operator_bit: UInt32[np.ndarray, " n_operators n_packed"],
        observables_phases: Int32[np.ndarray, " n_operators"],
        operator_phase: Int32[np.ndarray, " n_operators"],
        anticommuting_indices: UInt32[np.ndarray, " n_operators"],
    ) -> tuple[
        UInt32[np.ndarray, " n_operators n_packed"],
        Int32[np.ndarray, " n_operators"],
        UInt32[np.ndarray, " n_operators"],
        Bool[np.ndarray, " n_operators"],
    ]:
        new_pauli_operators, new_pauli_phases = compose_with(
            observables_bits[anticommuting_indices, :],
            operator_bit,
            observables_phases[anticommuting_indices],
            operator_phase,
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

        return new_pauli_operators, new_pauli_phases, new_pauli_indices, new_pauli_in_observables

    def add_new_paulis(
        self,
        observables_bits: UInt32[np.ndarray, " n_operators n_packed"],
        observables_phases: Int32[np.ndarray, " n_operators"],
        observables_coefficients: Complex64[jnp.ndarray, " n_operators"],
        new_bits: UInt32[np.ndarray, " n_operators n_packed"],
        new_phases: Int32[np.ndarray, " n_operators"],
        new_coeffs: Complex64[jnp.ndarray, " n_operators"],
        ind_to_add: UInt32[np.ndarray, " n_operators"],
    ) -> tuple[
        UInt32[np.ndarray, " new_n_operators n_packed"],
        Int32[np.ndarray, " new_n_operators"],
        Complex64[jnp.ndarray, " new_n_operators"],
    ]:
        new_bits, new_phases, new_coeffs = order_paulioperators(
            new_bits[ind_to_add, :],
            new_phases[ind_to_add],
            new_coeffs[ind_to_add],
        )
        # Insert new Paulis and return new array of coefficients.
        (
            observables_bits,
            observables_phases,
            observables_coefficients,
        ) = insert_paulioperators(
            bits=observables_bits,
            other_bits=new_bits,
            phases=observables_phases,
            other_phases=new_phases,
            coefficients=observables_coefficients,
            other_coefficients=new_coeffs,
            index=ind_to_add,
        )

        return observables_bits, observables_phases, observables_coefficients


def evaluate_expectation_value_zero_state(
    observables_bits: UInt32[np.ndarray, " n_operators n_packed"],
    observables_phases: Int32[np.ndarray, " n_operators"],
    observables_coefficients: Complex64[jnp.ndarray, " n_operators"],
) -> Complex64[jnp.ndarray, " 1"]:
    nonzero_pauli_indices = np.where(ztype(observables_bits))[0]
    return jnp.real(
        jnp.sum(
            observables_coefficients[nonzero_pauli_indices]
            * (-1j) ** observables_phases[nonzero_pauli_indices]
        ),
    )
