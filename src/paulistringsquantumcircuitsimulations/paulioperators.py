import jax
import jax.numpy as jnp
import numpy as np
import stim
from jaxtyping import Bool, Complex64, Int32, UInt32
from numba import set_num_threads

from paulistringsquantumcircuitsimulations.exceptions import SystemSizeError
from paulistringsquantumcircuitsimulations.utils import (
    anticommutation,
    bits_equal,
    delete_index_bits_and_phases,
    find_bit_index,
    insert_index_bits_and_phases,
    not_equal,
    pack_bits,
    update_phase,
    xor,
)

PauliString = str

set_num_threads(10)


def paulioperators_from_strings(
    n_qubits: int,
    paulistrings: list[PauliString],
    phases: list[int] | None = None,
    coefficients: list[complex] | list[float] | None = None,
) -> tuple[
    UInt32[np.ndarray, " n_operators n_packed"],
    Int32[np.ndarray, " n_operators"],
    Complex64[jnp.ndarray, " n_operators"],
]:
    if phases is None:
        phases = [0] * len(paulistrings)
    if coefficients is None:
        coefficients = [1.0 + 0.0j] * len(paulistrings)

    phases_: Int32[np.ndarray, " n_operators"] = np.array(phases, dtype=np.int32)
    coefficients_: Complex64[jnp.ndarray, " n_operators"] = jnp.array(coefficients, dtype=np.complex64)

    paulis: list[stim.PauliString] = [stim.PauliString(ps) for ps in paulistrings]
    xs, zs = zip(*[ps.to_numpy() for ps in paulis], strict=False)
    xs_ = np.array(xs, dtype=np.bool_)
    zs_ = np.array(zs, dtype=np.bool_)
    bits = np.hstack((pack_bits(zs_), pack_bits(xs_)))

    for ps in paulistrings:
        if len(ps) != n_qubits:
            raise SystemSizeError(len(ps), n_qubits)

    return (bits, phases_, coefficients_)


def compose_with(
    bits: UInt32[np.ndarray, " n_operators n_packed"],
    other_bit: UInt32[np.ndarray, " n_operators n_packed"],
    phases: Int32[np.ndarray, " n_operators"],
    other_phase: Int32[np.ndarray, " n_operators"],
) -> tuple[UInt32[np.ndarray, " n_operators n_packed"], Int32[np.ndarray, " n_operators"]]:
    n_packed = bits.shape[1] // 2
    phases = update_phase(
        bits[:, :n_packed],
        other_bit[:, n_packed:],
        phases,
        other_phase,
    )
    bits = xor(bits, other_bit)

    return bits, phases


def anticommutes(
    bits: UInt32[np.ndarray, " n_operators n_packed"],
    other_bit: UInt32[np.ndarray, " n_operators n_packed"],
) -> Bool[np.ndarray, " n_operators"]:
    n_packed = bits.shape[1] // 2
    a_dot_b = anticommutation(bits[:, n_packed:], other_bit[:, :n_packed])
    b_dot_a = anticommutation(bits[:, :n_packed], other_bit[:, n_packed:])
    result: Bool[np.ndarray, " n_operators"] = not_equal(a_dot_b, b_dot_a)
    return result


def find_paulioperators_indices(
    bits: UInt32[np.ndarray, " n_operators n_packed"],
    other_bits: UInt32[np.ndarray, " n_other_operators n_packed"],
) -> UInt32[np.ndarray, " n_other_operators"]:
    indices: UInt32[np.ndarray, " n_other_operators"] = find_bit_index(bits, other_bits)
    return indices


def find_paulioperators(
    bits: UInt32[np.ndarray, " n_operators n_packed"],
    other_bits: UInt32[np.ndarray, " n_operators n_packed"],
    index: UInt32[np.ndarray, " n_operators"],
) -> Bool[np.ndarray, " n_operators"]:
    (n_operators, _) = bits.shape
    result: Bool[np.ndarray, " n_operators"] = bits_equal(
        bits[index % n_operators, :],
        other_bits,
    )
    return result


def ztype(
    bits: UInt32[np.ndarray, " n_operators n_packed"],
    index: UInt32[np.ndarray, " n_operators-"] | None = None,
) -> Bool[np.ndarray, " n_operators"] | Bool[np.ndarray, " n_operators-"]:
    n_packed = bits.shape[1] // 2
    if index is None:
        result: Bool[np.ndarray, " n_operators"] = np.logical_not(np.any(bits[:, n_packed:], axis=1))
        return result
    result_: Bool[np.ndarray, " n_operators-"] = np.logical_not(np.any(bits[index, n_packed:], axis=1))
    return result_


def insert_paulioperators(
    bits: UInt32[np.ndarray, " n_operators n_packed"],
    other_bits: UInt32[np.ndarray, " n_operators n_packed"],
    phases: Int32[np.ndarray, " n_operators"],
    other_phases: Int32[np.ndarray, " n_operators"],
    coefficients: Complex64[jnp.ndarray, " n_operators"],
    other_coefficients: Complex64[jnp.ndarray, " n_operators"],
    index: UInt32[np.ndarray, " n_operators"],
) -> tuple[
    UInt32[np.ndarray, " new_n_operators n_packed"],
    Int32[np.ndarray, " new_n_operators"],
    Complex64[jnp.ndarray, " new_n_operators"],
]:
    index = find_paulioperators_indices(bits, other_bits)
    bits, phases = insert_index_bits_and_phases(
        bits,
        other_bits,
        phases,
        other_phases,
        index,
    )
    coefficients = insert_index_coefficients(
        coefficients,
        other_coefficients,
        index,
    )
    return bits, phases, coefficients


def delete_paulioperators(
    bits: UInt32[np.ndarray, " n_operators n_packed"],
    phases: Int32[np.ndarray, " n_operators"],
    coefficients: Complex64[jnp.ndarray, " n_operators"],
    index: UInt32[np.ndarray, " n_operators"],
) -> tuple[
    UInt32[np.ndarray, " new_n_operators n_packed"],
    Int32[np.ndarray, " new_n_operators"],
    Complex64[jnp.ndarray, " new_n_operators"],
]:
    bits, phases = delete_index_bits_and_phases(
        bits,
        phases,
        index,
    )
    coefficients = delete_index_coefficients(
        coefficients,
        index,
    )
    return bits, phases, coefficients


def order_paulioperators(
    bits: UInt32[np.ndarray, " n_operators n_packed"],
    phases: Int32[np.ndarray, " n_operators"],
    coefficients: Complex64[jnp.ndarray, " n_operators"],
) -> tuple[
    UInt32[np.ndarray, " n_operators n_packed"],
    Int32[np.ndarray, " n_operators"],
    Complex64[jnp.ndarray, " n_operators"],
]:
    two_n_packed = bits.shape[1]
    indices = np.lexsort([bits[:, j] for j in reversed(range(two_n_packed))])
    bits = bits[indices]
    phases = phases[indices]
    coefficients = coefficients[indices]
    return (bits, phases, coefficients)


def overlap(
    bits: UInt32[np.ndarray, " n_operators n_packed"],
    other_bits: UInt32[np.ndarray, " n_operators n_packed"],
    phases: Int32[np.ndarray, " n_operators"],
    other_phases: Int32[np.ndarray, " n_operators"],
    coefficients: Complex64[jnp.ndarray, " n_operators"],
    other_coefficients: Complex64[jnp.ndarray, " n_operators"],
) -> Complex64[jnp.ndarray, " 1"]:
    index = find_paulioperators_indices(bits, other_bits)
    pauli_found = find_paulioperators(bits, other_bits, index=index)
    index_found = index[pauli_found]
    return jnp.sum(
        coefficients[index_found]
        * jnp.conj(other_coefficients[pauli_found])
        * (-1j) ** (phases[index_found] - other_phases[pauli_found])
    )


@jax.jit
def update_coeffs(
    coeffs1: Complex64[jnp.ndarray, " n_operators"],
    coeffs2: Complex64[jnp.ndarray, " n_operators"],
    c: jnp.float32,
    s: jnp.float32,
    p1: jnp.int32,
    p2: jnp.int32,
    index1: jnp.uint32,
    index_exists: jnp.int32,
) -> Complex64[jnp.ndarray, " n_operators"]:
    coeffs = coeffs2 * (index_exists * (1j) * s * (-1j) ** (p2 - p1))
    return coeffs1.at[index1].set(jnp.take(coeffs1, index1) * c + coeffs)


@jax.jit
def coefs_lt_threshold(
    coefs: Complex64[jnp.ndarray, " n_operators"],
    threshold: float,
) -> Bool[jnp.ndarray, " n_operators"]:
    return jnp.abs(coefs) < threshold


@jax.jit
def coefs_gt_threshold(
    coefs: Complex64[jnp.ndarray, " n_operators"],
    threshold: float,
) -> Bool[jnp.ndarray, " n_operators"]:
    return jnp.abs(coefs) >= threshold


@jax.jit
def coefs_gt_threshold_and_not_logic(
    coefs: Complex64[jnp.ndarray, " n_operators"],
    threshold: float,
    logic: Bool[jnp.ndarray, " n_operators"],
) -> Bool[jnp.ndarray, " n_operators"]:
    return (jnp.abs(coefs) >= threshold) & (~logic)


@jax.jit
def insert_index_coefficients(
    coefficients: Complex64[jnp.ndarray, " n_operators"],
    other_coefficients: Complex64[jnp.ndarray, " n_operators"],
    index: UInt32[jnp.ndarray, " n_operators"],
) -> Complex64[jnp.ndarray, " new_n_operators"]:
    n_operators = coefficients.shape[0]
    n_other = other_coefficients.shape[0]
    new_size = n_operators + n_other

    dtype = coefficients.dtype

    res = jnp.zeros(new_size, dtype=dtype)

    insert_idx = index + jnp.arange(n_other)
    res = res.at[insert_idx].set(other_coefficients)

    full_idx = jnp.arange(new_size)
    keep_mask = ~jnp.isin(full_idx, insert_idx)
    keep_idx = jnp.nonzero(keep_mask, size=n_operators)[0]

    return res.at[keep_idx].set(coefficients)


@jax.jit
def delete_index_coefficients(
    coefficients: Complex64[jnp.ndarray, " n_operators"],
    index: UInt32[jnp.ndarray, " n_operators"],
) -> Complex64[jnp.ndarray, " new_n_operators"]:
    n_operators = coefficients.shape[0]
    full_indices = jnp.arange(n_operators)
    keep_mask = ~jnp.isin(full_indices, index)
    keep_indices = jnp.nonzero(keep_mask, size=n_operators)[0]
    return coefficients[keep_indices]
