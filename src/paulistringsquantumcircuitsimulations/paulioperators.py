from functools import partial

import jax
import jax.numpy as jnp
import stim
from jaxtyping import Bool, Complex128, Int64, UInt64

from paulistringsquantumcircuitsimulations.exceptions import SystemSizeError

PauliString = str


@jax.jit
def pack_bits(bool_jnp: Bool[jnp.ndarray, "n_op n_qubits"]) -> UInt64[jnp.ndarray, "n_op n_packed"]:
    """Pack boolean array into uint64 array.

    Args:
        bool_jnp: Bool[jnp.ndarray, "n_op n_qubits"]
            The boolean array to pack.

    Returns:
        UInt64[jnp.ndarray, "n_op n_packed"]: The packed boolean array.

    """
    (n_op, n_qubits) = bool_jnp.shape
    n_packed = (n_qubits + 63) // 64
    res = jnp.zeros((n_op, n_packed), dtype=jnp.uint64)

    blocks = jnp.arange(n_qubits) // 64
    positions = jnp.arange(n_qubits) % 64

    bit_values = (bool_jnp * (1 << positions)).astype(jnp.uint64)

    def update_row(
        res_row: Bool[jnp.ndarray, " n_packed"],
        bit_row: Bool[jnp.ndarray, " n_qubits"],
        blocks: Int64[jnp.ndarray, " n_qubits"],
    ) -> Bool[jnp.ndarray, " n_packed"]:
        return res_row.at[blocks].add(bit_row)

    return jax.vmap(update_row, in_axes=[0, 0, None])(res, bit_values, blocks)


@jax.jit
def find_bit_index(
    bits: UInt64[jnp.ndarray, "n_op n_packed"],
    other_bits: UInt64[jnp.ndarray, "n_op_others n_packed"],
) -> UInt64[jnp.ndarray, " n_op_others"]:
    """Find the indices of the Others Pauli operators in the PauliOperators.

    Args:
        bits: UInt64[jnp.ndarray, "n_op n_packed"]
            The bits of the Pauli operators.
        other_bits: UInt64[jnp.ndarray, "n_op_others n_packed"]
            The bits of the Pauli operators to find.

    Returns:
        UInt64[jnp.ndarray, " n_op_others"]: The indices of the Others Pauli operators in the PauliOperators.

    """
    (n_op, n_packed) = bits.shape
    (n_op_others, _) = other_bits.shape

    def search_single(other: UInt64[jnp.ndarray, " n_packed"]) -> UInt64[jnp.ndarray, " 1"]:
        """Find the first index where `bits` is greater than or equal to `other`."""
        mask = jnp.all(bits >= other, axis=1)
        valid_rows = jnp.where(mask, jnp.arange(n_op), n_op)
        return jnp.min(valid_rows)

    return jax.vmap(search_single, in_axes=[0])(other_bits)


@jax.jit
def bits_equal(
    bits: UInt64[jnp.ndarray, "n_op n_packed"],
    other_bits: UInt64[jnp.ndarray, "n_op_others n_packed"],
) -> Bool[jnp.ndarray, " n_op_others"]:
    """Compare bits arrays element-wise.

    Args:
        bits: UInt64[jnp.ndarray, "n_op n_packed"]
            First bits array.
        other_bits: UInt64[jnp.ndarray, "n_op_others n_packed"]
            Second bits array.

    Returns:
        Bool[jnp.ndarray, "n_op_others"]: Boolean array indicating where bits are equal.

    """
    return jnp.all(bits == other_bits, axis=1)


def insert_index(
    bits: UInt64[jnp.ndarray, "n_op n_packed"],
    other_bits: UInt64[jnp.ndarray, "n_op_others n_packed"],
    signs: Complex128[jnp.ndarray, " n_op"],
    other_signs: Complex128[jnp.ndarray, " n_op_others"],
    coefficients: Complex128[jnp.ndarray, " n_op"],
    other_coefficients: Complex128[jnp.ndarray, " n_op_others"],
    index: UInt64[jnp.ndarray, " n_op_others"],
) -> tuple[
    UInt64[jnp.ndarray, " n_op_new n_packed"],
    Complex128[jnp.ndarray, " n_op_new"],
    Complex128[jnp.ndarray, " n_op_new"],
]:
    """Insert `other_bits` into `bits` at positions `index`, updating `signs` and `coefficients` accordingly.

    Args:
        bits: UInt64[jnp.ndarray, "n_op n_packed"]
            The bits of the Pauli operators.
        other_bits: UInt64[jnp.ndarray, "n_op_others n_packed"]
            The bits of the Pauli operators to insert.
        signs: Complex128[jnp.ndarray, " n_op"]
            The signs of the Pauli operators.
        other_signs: Complex128[jnp.ndarray, " n_op_others"]
            The signs of the Pauli operators to insert.
        coefficients: Complex128[jnp.ndarray, " n_op"]
            The coefficients of the Pauli operators.
        other_coefficients: Complex128[jnp.ndarray, " n_op_others"]
            The coefficients of the Pauli operators to insert.
        index: UInt64[jnp.ndarray, " n_op_others"]
            The indices of the Pauli operators to insert.

    Returns:
        tuple[
            UInt64[jnp.ndarray, " n_op_new n_packed"],
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


def delete_index(
    bits: UInt64[jnp.ndarray, "n_op n_packed"],
    signs: Complex128[jnp.ndarray, " n_op"],
    coefficients: Complex128[jnp.ndarray, " n_op"],
    index: UInt64[jnp.ndarray, " n_op_others"],
) -> tuple[
    UInt64[jnp.ndarray, "n_op_new n_packed"],
    Complex128[jnp.ndarray, " n_op_new"],
    Complex128[jnp.ndarray, " n_op_new"],
]:
    """Delete elements from `bits`, `signs`, `coefficients` at given `index`.

    Args:
        bits: UInt64[jnp.ndarray, "n_op n_packed"]
            The bits of the Pauli operators.
        signs: Complex128[jnp.ndarray, " n_op"]
            The signs of the Pauli operators.
        coefficients: Complex128[jnp.ndarray, " n_op"]
            The coefficients of the Pauli operators.
        index: UInt64[jnp.ndarray, " n_op_others"]
            The indices of the Pauli operators to delete.

    Returns:
        tuple[
            UInt64[jnp.ndarray, "n_op_new n_packed"],
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


@jax.jit
def count_set_bits(n: jnp.uint64) -> jnp.uint32:
    """Count the number of set bits in a 64-bit integer using bitwise operations."""
    n = n - ((n >> 1) & 0x5555555555555555)
    n = (n & 0x3333333333333333) + ((n >> 2) & 0x3333333333333333)
    n = (n + (n >> 4)) & 0x0F0F0F0F0F0F0F0F
    n = n + (n >> 8)
    n = n + (n >> 16)
    n = n + (n >> 32)
    return n & 0x7F


@jax.jit
def count_nonzero(bits: UInt64[jnp.ndarray, "n_op n_packed"]) -> jnp.uint64:
    """Count the total number of set bits in `bits`."""
    return jnp.sum(jax.vmap(count_set_bits)(bits))


@jax.jit
def anticommutation(
    bits: UInt64[jnp.ndarray, "n_op n_packed"],
    other_bit: UInt64[jnp.ndarray, "1 n_packed"],
) -> UInt64[jnp.ndarray, " n_op"]:
    return jax.vmap(lambda row: jnp.mod(count_nonzero(jnp.bitwise_and(row, other_bit[0])), 2))(bits)


@jax.jit
def new_sign(
    bits: UInt64[jnp.ndarray, "n_op n_packed"],
    other_bit: UInt64[jnp.ndarray, "1 n_packed"],
    signs: Complex128[jnp.ndarray, " n_op"],
    other_sign: Complex128[jnp.ndarray, " 1"],
) -> Complex128[jnp.ndarray, " n_op"]:
    """Update signs of Pauli operators during composition.

    Args:
        bits: UInt64[jnp.ndarray, "n_op n_packed"]
            The bits of the first Pauli operator.
        other_bit: UInt64[jnp.ndarray, "1 n_packed"]
            The bits of the second Pauli operator.
        signs: Complex128[jnp.ndarray, "n_op"]
            The signs of the first Pauli operator.
        other_sign: Complex128[jnp.ndarray, " 1"]
            The sign of the second Pauli operator.

    """

    def compute_sign(
        row: UInt64[jnp.ndarray, " n_packed"],
        s1: jnp.complex128,
    ) -> jnp.complex128:
        n_common = jnp.count_nonzero(jnp.bitwise_and(row, other_bit[0]))
        return s1 * other_sign * ((-1j) ** (2 * n_common))

    result: Complex128[jnp.ndarray, " n_op"] = jax.vmap(compute_sign, in_axes=[0, 0])(bits, signs).flatten()
    return result


@jax.jit
def not_equal(
    bits: UInt64[jnp.ndarray, " n_op"],
    other_bits: UInt64[jnp.ndarray, " n_op"],
) -> Bool[jnp.ndarray, " n_op"]:
    return bits != other_bits


@jax.jit
def xor(
    bits: UInt64[jnp.ndarray, "n_op n_packed"],
    other_bit: UInt64[jnp.ndarray, "1 n_packed"],
) -> UInt64[jnp.ndarray, " n_op n_packed"]:
    return jnp.bitwise_xor(bits, other_bit)


@partial(jax.jit, static_argnums=(4,))
def compose_with(
    bits: UInt64[jnp.ndarray, "n_op n_packed"],
    other_bit: UInt64[jnp.ndarray, "1 n_packed"],
    signs: Complex128[jnp.ndarray, " n_op"],
    other_sign: Complex128[jnp.ndarray, " 1"],
    n_qubits: int,
) -> tuple[UInt64[jnp.ndarray, " n_op n_packed"], Complex128[jnp.ndarray, " n_op"]]:
    """Composes all Paulis in 'self' with the Pauli (only one Pauli allowed) in 'other'.

    Args:
        bits: UInt64[jnp.ndarray, "n_op n_packed"]
            The bits of the first Pauli operator.
        other_bit: UInt64[jnp.ndarray, "1 n_packed"]
            The bit of the second Pauli operator.
        signs: Complex128[jnp.ndarray, " n_op"]
            The sign of the first Pauli operator.
        other_sign: Complex128[jnp.ndarray, " 1"]
            The sign of the second Pauli operator.
        n_qubits: int
            The number of qubits.

    Returns:
        tuple[UInt64[jnp.ndarray, " n_op n_packed"], Complex128[jnp.ndarray, " n_op"]]:
            The bits and signs of the Pauli operators.

    """
    nq = (n_qubits + 63) // 64
    signs = new_sign(
        jax.lax.dynamic_slice(bits, (0, 0), (bits.shape[0], nq)),
        jax.lax.dynamic_slice(other_bit, (0, nq), (other_bit.shape[0], other_bit.shape[1])),
        signs,
        other_sign,
    )
    bits = xor(bits, other_bit)

    return bits, signs


def paulioperators_from_strings(
    paulistrings: list[PauliString],
    n_qubits: int,
    signs: Complex128[jnp.ndarray, " n_op"] | list[complex] | None = None,
    coefficients: Complex128[jnp.ndarray, " n_op"] | list[complex] | None = None,
) -> tuple[
    UInt64[jnp.ndarray, "n_op n_packed"],
    Complex128[jnp.ndarray, " n_op"],
    Complex128[jnp.ndarray, " n_op"],
]:
    """Create a PauliOperators from a list of Pauli strings and a list of signs and coefficients.

    Args:
        paulistrings: list[str]
            A list of Pauli strings.
        n_qubits: int
            The number of qubits.
        signs: list[complex]
            The list of signs of the Pauli operators.
        coefficients: list[complex]
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


def order_paulioperators(
    bits: UInt64[jnp.ndarray, "n_op n_packed"],
    signs: Complex128[jnp.ndarray, " n_op"],
    coefficients: Complex128[jnp.ndarray, " n_op"],
) -> tuple[
    UInt64[jnp.ndarray, "n_op n_packed"],
    Complex128[jnp.ndarray, " n_op"],
    Complex128[jnp.ndarray, " n_op"],
]:
    (n_operators, nq_orders) = bits.shape
    indices = jnp.lexsort([bits[:, j] for j in reversed(range(nq_orders))])
    bits = bits[indices]
    signs = signs[indices]
    coefficients = coefficients[indices]
    return (bits, signs, coefficients)


def find_paulioperators_indices(
    bits: UInt64[jnp.ndarray, "n_op n_packed"],
    other_bits: UInt64[jnp.ndarray, "n_op_others n_packed"],
) -> UInt64[jnp.ndarray, " n_op_others"]:
    """Find the indices of the other_bits in the bits.

    Args:
        bits: UInt64[jnp.ndarray, "n_op n_packed"]
            The bits of the Pauli operators.
        other_bits: UInt64[jnp.ndarray, "n_op_others n_packed"]
            The bits of the Pauli operators to find the indices of.

    Returns:
        UInt64[jnp.ndarray, " n_op_others"]:
            The indices of the other_bits in the bits.

    """
    result: UInt64[jnp.ndarray, " n_op_others"] = find_bit_index(bits, other_bits)
    return result


def find_paulioperators(
    bits: UInt64[jnp.ndarray, "n_op n_packed"],
    other_bits: UInt64[jnp.ndarray, "n_op_others n_packed"],
    index: UInt64[jnp.ndarray, " n_op_others"] | None = None,
) -> Bool[jnp.ndarray, " n_op_others"]:
    """Find the indices of the other_bits in the bits.

    Args:
        bits: UInt64[jnp.ndarray, "n_op n_packed"]
            The bits of the Pauli operators.
        other_bits: UInt64[jnp.ndarray, "n_op_others n_packed"]
            The bits of the Pauli operators to find.
        index: UInt64[jnp.ndarray, " n_op_others"] | None
            The indices of the Pauli operators to find.

    Returns:
        Bool[jnp.ndarray, " n_op_others"]:
            The boolean array indicating if the other_bits are in the bits.

    """
    if index is None:
        index = find_paulioperators_indices(bits, other_bits)
    (n_operators, _) = bits.shape
    result: Bool[jnp.ndarray, " n_op_others"] = bits_equal(
        bits[index % n_operators, :],
        other_bits,
    )
    return result


def insert_paulioperators(
    bits: UInt64[jnp.ndarray, "n_op n_packed"],
    other_bits: UInt64[jnp.ndarray, "n_op_others n_packed"],
    signs: Complex128[jnp.ndarray, " n_op"],
    other_signs: Complex128[jnp.ndarray, " n_op_others"],
    coefficients: Complex128[jnp.ndarray, " n_op"],
    other_coefficients: Complex128[jnp.ndarray, " n_op_others"],
    index: UInt64[jnp.ndarray, " n_op_others"],
) -> tuple[
    UInt64[jnp.ndarray, " n_op_new n_packed"],
    Complex128[jnp.ndarray, " n_op_new"],
    Complex128[jnp.ndarray, " n_op_new"],
]:
    """Insert paulioperators given by other_bits into bits at index.

    Args:
        bits: UInt64[jnp.ndarray, "n_op n_packed"]
            The bits of the Pauli operators.
        other_bits: UInt64[jnp.ndarray, "n_op_others n_packed"]
            The PauliOperators to insert.
        signs: Complex128[jnp.ndarray, " n_op"]
            The signs of the Pauli operators.
        other_signs: Complex128[jnp.ndarray, " n_op_others"]
            The signs of the Pauli operators to insert.
        coefficients: Complex128[jnp.ndarray, " n_op"]
            The coefficients of the Pauli operators.
        other_coefficients: Complex128[jnp.ndarray, " n_op_others"]
            The coefficients of the Pauli operators to insert.
        index: UInt64[jnp.ndarray, " n_op_others"]
            The indices of the Pauli operators to insert.

    Returns:
        tuple[
            UInt64[jnp.ndarray, " n_op_new n_packed"],
            Complex128[jnp.ndarray, " n_op_new"],
            Complex128[jnp.ndarray, " n_op_new"],
        ]:
            The bits, signs, and coefficients of the Pauli operators

    """
    index = find_paulioperators_indices(bits, other_bits)
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


def delete_paulioperators(
    bits: UInt64[jnp.ndarray, "n_op n_packed"],
    signs: Complex128[jnp.ndarray, " n_op"],
    coefficients: Complex128[jnp.ndarray, " n_op"],
    index: UInt64[jnp.ndarray, " n_op_others"],
) -> tuple[
    UInt64[jnp.ndarray, "n_op n_packed"],
    Complex128[jnp.ndarray, " n_op"],
    Complex128[jnp.ndarray, " n_op"],
]:
    """Delete the Pauli operators at the given indices.

    Args:
        bits: UInt64[jnp.ndarray, "n_op n_packed"]
            The bits of the Pauli operators.
        signs: Complex128[jnp.ndarray, " n_op"]
            The signs of the Pauli operators.
        coefficients: Complex128[jnp.ndarray, " n_op"]
            The coefficients of the Pauli operators.
        index: UInt64[jnp.ndarray, " n_op_others"]
            The indices of the Pauli operators to delete.

    Returns:
        tuple[
            UInt64[jnp.ndarray, "n_op n_packed"],
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


def anticommutes(
    bits: UInt64[jnp.ndarray, "n_op n_packed"],
    other_bit: UInt64[jnp.ndarray, "1 n_packed"],
    n_qubits: int,
) -> Bool[jnp.ndarray, " n_op"]:
    """Check if the Pauli operators anticommute.

    Args:
        bits: UInt64[jnp.ndarray, "n_op n_packed"]
            The bits of the Pauli operators.
        other_bit: UInt64[jnp.ndarray, "1 n_packed"]
            The bit of the Pauli operator to check if it anticommutes with the Pauli operators in bits.
        n_qubits: int
            The number of qubits.

    Returns:
        Bool[jnp.ndarray, " n_op"]: Array indicating which operators anticommute.

    """
    nq = (n_qubits + 63) // 64
    a_dot_b = anticommutation(bits[:, nq:], other_bit[:, nq:])
    b_dot_a = anticommutation(bits[:, :nq], other_bit[:, :nq])
    result: Bool[jnp.ndarray, " n_op"] = not_equal(a_dot_b, b_dot_a)
    return result


def ztype(
    bits: UInt64[jnp.ndarray, "n_op n_packed"],
    n_qubits: int,
    index: UInt64[jnp.ndarray, " n_op_others"] | None = None,
) -> Bool[jnp.ndarray, " n_op_others"]:
    """Return logical array indicating whether a Pauli in self is composed only of Z or identity Pauli.

    Args:
        bits: UInt64[jnp.ndarray, "n_op n_packed"]
            The bits of the Pauli operators.
        n_qubits: int
            The number of qubits.
        index: UInt64[jnp.ndarray, " n_op_others"] | None
            The indices of the Pauli operators to check, if needed.

    Returns:
        Bool[jnp.ndarray, " n_op_others"]: Array indicating which operators are Z or identity.

    """
    nq = (n_qubits + 63) // 64
    if index is None:
        result: Bool[jnp.ndarray, " n_op"] = jnp.logical_not(jnp.any(bits[:, nq:], axis=1))
    else:
        result = jnp.logical_not(jnp.any(bits[index, nq:], axis=1))
    return result
