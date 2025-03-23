from collections.abc import Callable

import jax.numpy as jnp
import stim
from jaxtyping import Bool, Complex128, Float64, UInt64

from paulistringsquantumcircuitsimulations.exceptions import SystemSizeError
from paulistringsquantumcircuitsimulations.exported import (
    anticommutation_exported,
    new_sign_exported,
    not_equal_exported,
    pack_bits_exported,
    xor_exported,
    ztype_bool_exported,
)
from paulistringsquantumcircuitsimulations.utils import PauliString


def paulioperators_from_strings(
    n_qubits: int,
    paulistrings: list[PauliString],
    signs: Complex128[jnp.ndarray, " n_op"] | list[complex] | None = None,
    coefficients: Complex128[jnp.ndarray, " n_op"]
    | Float64[jnp.ndarray, " n_op"]
    | list[complex]
    | list[float]
    | None = None,
    pack_bits: Callable | None = None,
) -> tuple[
    UInt64[jnp.ndarray, "n_op 2n_packed"],
    Complex128[jnp.ndarray, " n_op"],
    Complex128[jnp.ndarray, " n_op"],
]:
    """Create a PauliOperators from a list of Pauli strings and a list of signs and coefficients.

    Args:
        paulistrings (list[str]):
            A list of Pauli strings.
        n_qubits (int):
            The number of qubits.
        pack_bits (Callable):
            The function to pack the bits.
        signs (list[complex]):
            The list of signs of the Pauli operators.
        coefficients (list[complex] | list[float] | None = None):
            The list of coefficients of the Pauli operators.

    Returns:
        tuple[
            UInt64[jnp.ndarray, "n_op n_packed"],
            Complex128[jnp.ndarray, " n_op"],
            Complex128[jnp.ndarray, " n_op"],
        ]:
            The bits, signs, and coefficients of the Pauli operators.

    """
    if signs is None:
        signs = [1.0 + 0.0j] * len(paulistrings)
    if coefficients is None:
        coefficients = [1.0 + 0.0j] * len(paulistrings)
    if pack_bits is None:
        pack_bits = pack_bits_exported(n_qubits)

    signs_: Complex128[jnp.ndarray, " n_op"] = jnp.array(signs, dtype=jnp.complex128)
    coefficients_: Complex128[jnp.ndarray, " n_op"] = jnp.array(coefficients, dtype=jnp.complex128)

    paulis: list[stim.PauliString] = [stim.PauliString(ps) for ps in paulistrings]
    xs, zs = zip(*[ps.to_numpy() for ps in paulis], strict=False)
    xs_ = jnp.array(xs, dtype=jnp.bool_)
    zs_ = jnp.array(zs, dtype=jnp.bool_)
    bits = jnp.hstack((pack_bits(zs_), pack_bits(xs_)))

    for ps in paulistrings:
        if len(ps) != n_qubits:
            raise SystemSizeError(len(ps), n_qubits)

    return (bits, signs_, coefficients_)


def compose_with(
    n_qubits: int,
    bits: UInt64[jnp.ndarray, "n_op 2n_packed"],
    other_bit: UInt64[jnp.ndarray, "1 2n_packed"],
    signs: Complex128[jnp.ndarray, " n_op"],
    other_sign: Complex128[jnp.ndarray, " 1"],
    new_sign: Callable | None = None,
    xor: Callable | None = None,
) -> tuple[UInt64[jnp.ndarray, " n_op 2n_packed"], Complex128[jnp.ndarray, " n_op"]]:
    """Composes all Paulis in 'self' with the Pauli (only one Pauli allowed) in 'other'.

    Args:
        n_qubits (int):
            The number of qubits.
        bits (UInt64[jnp.ndarray, "n_op 2n_packed"]):
            The bits of the first Pauli operator.
        other_bit (UInt64[jnp.ndarray, "1 2n_packed"]):
            The bit of the second Pauli operator.
        signs (Complex128[jnp.ndarray, " n_op"]):
            The sign of the first Pauli operator.
        other_sign (Complex128[jnp.ndarray, " 1"]):
            The sign of the second Pauli operator.
        new_sign (Callable):
            The function to compute the new sign.
        xor (Callable):
            The function to compute the new bits.

    Returns:
        tuple[UInt64[jnp.ndarray, " n_op 2n_packed"], Complex128[jnp.ndarray, " n_op"]]:
            The bits and signs of the Pauli operators.

    """
    if new_sign is None:
        new_sign = new_sign_exported(n_qubits)
    if xor is None:
        xor = xor_exported(n_qubits)

    nq = (n_qubits + 63) // 64

    signs = new_sign(
        bits[:, nq:],
        other_bit[:, :nq],
        signs,
        other_sign,
    )
    bits = xor(bits, other_bit)

    return bits, signs


def anticommutes(
    n_qubits: int,
    bits: UInt64[jnp.ndarray, "n_op n_packed"],
    other_bit: UInt64[jnp.ndarray, "1 n_packed"],
    anticommutation: Callable | None = None,
    not_equal: Callable | None = None,
) -> Bool[jnp.ndarray, " n_op"]:
    """Check if the Pauli operators anticommute.

    Args:
        n_qubits (int):
            The number of qubits.
        bits (UInt64[jnp.ndarray, "n_op n_packed"]):
            The bits of the Pauli operators.
        other_bit (UInt64[jnp.ndarray, "1 n_packed"]):
            The bit of the Pauli operator to check if it anticommutes with the Pauli operators in bits.
        anticommutation (Callable):
            The function to compute the anticommutation.
        not_equal (Callable):
            The function to check if the bits are not equal.

    Returns:
        Bool[jnp.ndarray, " n_op"]: Array indicating which operators anticommute.

    """
    if anticommutation is None:
        anticommutation = anticommutation_exported(n_qubits)
    if not_equal is None:
        not_equal = not_equal_exported()

    n_packed = (n_qubits + 63) // 64
    a_dot_b = anticommutation(bits[:, n_packed:], other_bit[:, :n_packed])
    b_dot_a = anticommutation(bits[:, :n_packed], other_bit[:, n_packed:])
    result: Bool[jnp.ndarray, " n_op"] = not_equal(a_dot_b, b_dot_a).astype(jnp.bool_)
    return result


def find_paulioperators_indices(
    bits: UInt64[jnp.ndarray, "n_op 2n_packed"],
    other_bits: UInt64[jnp.ndarray, "n_op_others 2n_packed"],
    find_bit_index: Callable,
) -> UInt64[jnp.ndarray, " n_op_others"]:
    """Find the indices of the other_bits in the bits.

    Args:
        bits (UInt64[jnp.ndarray, "n_op 2n_packed"]):
            The bits of the Pauli operators.
        other_bits (UInt64[jnp.ndarray, "n_op_others 2n_packed"]):
            The bits of the Pauli operators to find the indices of.
        find_bit_index (Callable):
            The function to find the index of the other_bits in the bits.

    Returns:
        UInt64[jnp.ndarray, " n_op_others"]:
            The indices of the other_bits in the bits.

    """
    result: UInt64[jnp.ndarray, " n_op_others"] = find_bit_index(bits, other_bits)
    return result


def find_paulioperators(
    bits: UInt64[jnp.ndarray, "n_op 2n_packed"],
    other_bits: UInt64[jnp.ndarray, "n_op_others 2n_packed"],
    index: UInt64[jnp.ndarray, " n_op_others"],
    bits_equal: Callable,
) -> Bool[jnp.ndarray, " n_op_others"]:
    """Find the indices of the other_bits in the bits.

    Args:
        bits (UInt64[jnp.ndarray, "n_op 2n_packed"]):
            The bits of the Pauli operators.
        other_bits (UInt64[jnp.ndarray, "n_op_others 2n_packed"]):
            The bits of the Pauli operators to find.
        bits_equal (Callable):
            The function to check if the bits are equal.
        find_bit_index (Callable):
            The function to find the index of the other_bits in the bits.
        index (UInt64[jnp.ndarray, " n_op_others"] | None):
            The indices of the Pauli operators to find.

    Returns:
        Bool[jnp.ndarray, " n_op_others"]:
            The boolean array indicating if the other_bits are in the bits.

    """
    (n_operators, _) = bits.shape
    result: Bool[jnp.ndarray, " n_op_others"] = bits_equal(
        bits[index % n_operators, :],
        other_bits,
    )
    return result


def ztype(
    n_qubits: int,
    bits: UInt64[jnp.ndarray, "n_op 2n_packed"],
    index: UInt64[jnp.ndarray, " n_op_others"] | None = None,
    ztype_bool: Callable | None = None,
) -> Bool[jnp.ndarray, " n_op_others"]:
    """Return logical array indicating whether a Pauli in self is composed only of Z or identity Pauli.

    Args:
        bits (UInt64[jnp.ndarray, "n_op n_packed"]):
            The bits of the Pauli operators.
        n_qubits (int):
            The number of qubits.
        ztype_bool (Callable):
            The function to check if the bits are Z or identity.
        index (UInt64[jnp.ndarray, " n_op_others"] | None):
            The indices of the Pauli operators to check, if needed.

    Returns:
        Bool[jnp.ndarray, " n_op_others"]: Array indicating which operators are Z or identity.

    """
    if ztype_bool is None:
        ztype_bool = ztype_bool_exported(n_qubits)

    n_packed = (n_qubits + 63) // 64
    if index is None:
        result: Bool[jnp.ndarray, " n_op"] = ztype_bool(bits[:, n_packed:])
    else:
        result = ztype_bool(bits[index, n_packed:])
    return result


def insert_index(
    bits: UInt64[jnp.ndarray, "n_op 2n_packed"],
    other_bits: UInt64[jnp.ndarray, "n_op_others 2n_packed"],
    signs: Complex128[jnp.ndarray, " n_op"],
    other_signs: Complex128[jnp.ndarray, " n_op_others"],
    coefficients: Complex128[jnp.ndarray, " n_op"],
    other_coefficients: Complex128[jnp.ndarray, " n_op_others"],
    index: UInt64[jnp.ndarray, " n_op_others"],
) -> tuple[
    UInt64[jnp.ndarray, " n_op_new 2n_packed"],
    Complex128[jnp.ndarray, " n_op_new"],
    Complex128[jnp.ndarray, " n_op_new"],
]:
    """Insert `other_bits` into `bits` at positions `index`, updating `signs` and `coefficients` accordingly.

    Args:
        bits (UInt64[jnp.ndarray, "n_op 2n_packed"]):
            The bits of the Pauli operators.
        other_bits (UInt64[jnp.ndarray, "n_op_others 2n_packed"]):
            The bits of the Pauli operators to insert.
        signs (Complex128[jnp.ndarray, " n_op"]):
            The signs of the Pauli operators.
        other_signs (Complex128[jnp.ndarray, " n_op_others"]):
            The signs of the Pauli operators to insert.
        coefficients (Complex128[jnp.ndarray, " n_op"]):
            The coefficients of the Pauli operators.
        other_coefficients (Complex128[jnp.ndarray, " n_op_others"]):
            The coefficients of the Pauli operators to insert.
        index (UInt64[jnp.ndarray, " n_op_others"]):
            The indices of the Pauli operators to insert.

    Returns:
        tuple[
            UInt64[jnp.ndarray, " n_op_new 2n_packed"],
            Complex128[jnp.ndarray, " n_op_new"],
            Complex128[jnp.ndarray, " n_op_new"],
        ]:
            The bits, signs, and coefficients of the Pauli operators.

    """
    (n_op, n_packed) = bits.shape
    n_op_others = len(other_bits)
    new_size = n_op + n_op_others

    res_bits = jnp.zeros((new_size, n_packed), dtype=jnp.uint64)
    res_signs = jnp.zeros(new_size, dtype=jnp.complex128)
    res_coefficients = jnp.zeros(new_size, dtype=jnp.complex128)

    insert_pos = index + jnp.arange(n_op_others)

    res_bits = res_bits.at[insert_pos].set(other_bits)
    res_signs = res_signs.at[insert_pos].set(other_signs)
    res_coefficients = res_coefficients.at[insert_pos].set(other_coefficients)

    all_indices = jnp.arange(new_size)
    mask = ~jnp.isin(all_indices, insert_pos)
    bits_indices = all_indices[mask]

    res_bits = res_bits.at[bits_indices].set(bits)
    res_signs = res_signs.at[bits_indices].set(signs)
    res_coefficients = res_coefficients.at[bits_indices].set(coefficients)

    return res_bits, res_signs, res_coefficients


def insert_paulioperators(
    bits: UInt64[jnp.ndarray, "n_op 2n_packed"],
    other_bits: UInt64[jnp.ndarray, "n_op_others 2n_packed"],
    signs: Complex128[jnp.ndarray, " n_op"],
    other_signs: Complex128[jnp.ndarray, " n_op_others"],
    coefficients: Complex128[jnp.ndarray, " n_op"],
    other_coefficients: Complex128[jnp.ndarray, " n_op_others"],
    index: UInt64[jnp.ndarray, " n_op_others"],
    find_bit_index: Callable,
) -> tuple[
    UInt64[jnp.ndarray, " n_op_new 2n_packed"],
    Complex128[jnp.ndarray, " n_op_new"],
    Complex128[jnp.ndarray, " n_op_new"],
]:
    """Insert paulioperators given by other_bits into bits at index.

    Args:
        bits (UInt64[jnp.ndarray, "n_op 2n_packed"]):
            The bits of the Pauli operators.
        other_bits (UInt64[jnp.ndarray, "n_op_others 2n_packed"]):
            The PauliOperators to insert.
        signs (Complex128[jnp.ndarray, " n_op"]):
            The signs of the Pauli operators.
        other_signs (Complex128[jnp.ndarray, " n_op_others"]):
            The signs of the Pauli operators to insert.
        coefficients (Complex128[jnp.ndarray, " n_op"]):
            The coefficients of the Pauli operators.
        other_coefficients (Complex128[jnp.ndarray, " n_op_others"]):
            The coefficients of the Pauli operators to insert.
        index (UInt64[jnp.ndarray, " n_op_others"]):
            The indices of the Pauli operators to insert.
        find_bit_index (Callable):
            The function to find the index of the other_bits in the bits.

    Returns:
        tuple[
            UInt64[jnp.ndarray, " n_op_new 2n_packed"],
            Complex128[jnp.ndarray, " n_op_new"],
            Complex128[jnp.ndarray, " n_op_new"],
        ]:
            The bits, signs, and coefficients of the Pauli operators

    """
    index = find_paulioperators_indices(bits, other_bits, find_bit_index=find_bit_index)
    bits, signs, coefficients = insert_index(
        bits,
        other_bits,
        signs,
        other_signs,
        coefficients,
        other_coefficients,
        index,
    )
    return bits, signs, coefficients


def delete_index(
    bits: UInt64[jnp.ndarray, "n_op 2n_packed"],
    signs: Complex128[jnp.ndarray, " n_op"],
    coefficients: Complex128[jnp.ndarray, " n_op"],
    index: UInt64[jnp.ndarray, " n_op_others"],
) -> tuple[
    UInt64[jnp.ndarray, "n_op_new 2n_packed"],
    Complex128[jnp.ndarray, " n_op_new"],
    Complex128[jnp.ndarray, " n_op_new"],
]:
    """Delete elements from `bits`, `signs`, `coefficients` at given `index`.

    Args:
        bits (UInt64[jnp.ndarray, "n_op 2n_packed"]):
            The bits of the Pauli operators.
        signs (Complex128[jnp.ndarray, " n_op"]):
            The signs of the Pauli operators.
        coefficients (Complex128[jnp.ndarray, " n_op"]):
            The coefficients of the Pauli operators.
        index (UInt64[jnp.ndarray, " n_op_others"]):
            The indices of the Pauli operators to delete.

    Returns:
        tuple[
            UInt64[jnp.ndarray, "n_op_new 2n_packed"],
            Complex128[jnp.ndarray, " n_op_new"],
            Complex128[jnp.ndarray, " n_op_new"],
        ]:
            The bits, signs, and coefficients of the Pauli operators.

    """
    (n_op, n_packed) = bits.shape

    mask = jnp.ones(n_op, dtype=jnp.bool_)
    mask = mask.at[index].set(False)

    remaining_indices = jnp.where(mask)[0]

    res_bits = bits[remaining_indices]
    res_signs = signs[remaining_indices]
    res_coefficients = coefficients[remaining_indices]

    return res_bits, res_signs, res_coefficients


def delete_paulioperators(
    bits: UInt64[jnp.ndarray, "n_op 2n_packed"],
    signs: Complex128[jnp.ndarray, " n_op"],
    coefficients: Complex128[jnp.ndarray, " n_op"],
    index: UInt64[jnp.ndarray, " n_op_others"],
) -> tuple[
    UInt64[jnp.ndarray, "n_op 2n_packed"],
    Complex128[jnp.ndarray, " n_op"],
    Complex128[jnp.ndarray, " n_op"],
]:
    """Delete the Pauli operators at the given indices.

    Args:
        bits (UInt64[jnp.ndarray, "n_op 2n_packed"]):
            The bits of the Pauli operators.
        signs (Complex128[jnp.ndarray, " n_op"]):
            The signs of the Pauli operators.
        coefficients (Complex128[jnp.ndarray, " n_op"]):
            The coefficients of the Pauli operators.
        index (UInt64[jnp.ndarray, " n_op_others"]):
            The indices of the Pauli operators to delete.

    Returns:
        tuple[
            UInt64[jnp.ndarray, "n_op 2n_packed"],
            Complex128[jnp.ndarray, " n_op"],
            Complex128[jnp.ndarray, " n_op"],
        ]:
            The bits, signs, and coefficients of the Pauli operators.

    """
    bits, signs, coefficients = delete_index(
        bits,
        signs,
        coefficients,
        index,
    )
    return bits, signs, coefficients


def order_paulioperators(
    bits: UInt64[jnp.ndarray, "n_op 2n_packed"],
    signs: Complex128[jnp.ndarray, " n_op"],
    coefficients: Complex128[jnp.ndarray, " n_op"],
) -> tuple[
    UInt64[jnp.ndarray, "n_op 2n_packed"],
    Complex128[jnp.ndarray, " n_op"],
    Complex128[jnp.ndarray, " n_op"],
]:
    (n_operators, nq_orders) = bits.shape
    indices = jnp.lexsort([bits[:, j] for j in reversed(range(nq_orders))])
    bits = bits[indices]
    signs = signs[indices]
    coefficients = coefficients[indices]
    return (bits, signs, coefficients)


def overlap(
    bits: UInt64[jnp.ndarray, "n_op 2n_packed"],
    other_bits: UInt64[jnp.ndarray, "n_op_others 2n_packed"],
    signs: Complex128[jnp.ndarray, " n_op"],
    other_signs: Complex128[jnp.ndarray, " n_op_others"],
    coefficients: Complex128[jnp.ndarray, " n_op"],
    other_coefficients: Complex128[jnp.ndarray, " n_op_others"],
    find_bit_index: Callable,
    bits_equal: Callable,
) -> Complex128[jnp.ndarray, " n_op_others"]:
    """Compute overlap of two Pauli sums as Tr[B^dag A] / N, where N is a normalization factor.

    Bits (A) and other (B) are both PauliRepresentation objects.

    Args:
        bits (UInt64[jnp.ndarray, "n_op 2n_packed"]):
            The bits of the Pauli operators.
        other_bits (UInt64[jnp.ndarray, "n_op_others 2n_packed"]):
            The bits of the Pauli operators to compute the overlap with.
        signs (Complex128[jnp.ndarray, " n_op"]):
            The signs of the Pauli operators.
        other_signs (Complex128[jnp.ndarray, " n_op_others"]):
            The signs of the Pauli operators to compute the overlap with.
        coefficients (Complex128[jnp.ndarray, " n_op"]):
            The coefficients of the Pauli operators.
        other_coefficients (Complex128[jnp.ndarray, " n_op_others"]):
            The coefficients of the Pauli operators to compute the overlap with.
        find_bit_index (Callable):
            The function to find the index of the other_bits in the bits.
        bits_equal (Callable):
            The function to check if the bits are equal.

    Returns:
        Complex128[jnp.ndarray, " n_op_others"]:
            The overlap of the Pauli operators.

    """
    index = find_paulioperators_indices(bits, other_bits, find_bit_index=find_bit_index)
    pauli_found = find_paulioperators(bits, other_bits, index=index, bits_equal=bits_equal)
    index_found = index[pauli_found]
    return jnp.sum(
        coefficients[index_found]
        * jnp.conj(other_coefficients[pauli_found])
        * signs[index_found]
        / other_signs[pauli_found]
    )
