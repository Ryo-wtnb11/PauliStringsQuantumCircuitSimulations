import jax
import jax.numpy as jnp
from jaxtyping import Bool, Complex128, Int64, UInt64

PauliString = str


def anticommutation(
    bits: UInt64[jnp.ndarray, "n_op n_packed"],
    other_bit: UInt64[jnp.ndarray, "1 n_packed"],
) -> UInt64[jnp.ndarray, " n_op"]:
    return jax.vmap(lambda row: jnp.mod(count_nonzero(jnp.bitwise_and(row, other_bit[0])), 2))(bits)


def bits_equal(
    bits: UInt64[jnp.ndarray, "n_op 2n_packed"],
    other_bits: UInt64[jnp.ndarray, "n_op_others 2n_packed"],
) -> Bool[jnp.ndarray, " n_op_others"]:
    """Compare bits arrays element-wise.

    Args:
        bits (UInt64[jnp.ndarray, "n_op 2n_packed"]):
            First bits array.
        other_bits (UInt64[jnp.ndarray, "n_op_others 2n_packed"]):
            Second bits array.

    Returns:
        Bool[jnp.ndarray, "n_op_others"]: Boolean array indicating where bits are equal.

    """
    return jnp.all(bits == other_bits, axis=1)


def count_nonzero(bits: UInt64[jnp.ndarray, "n_op n_packed"]) -> jnp.uint64:
    """Count the total number of set bits in `bits`."""
    return jnp.sum(jax.vmap(count_set_bits)(bits))


def find_bit_index(
    bits: UInt64[jnp.ndarray, "n_op 2n_packed"],
    other_bits: UInt64[jnp.ndarray, "n_op_others 2n_packed"],
) -> UInt64[jnp.ndarray, " n_op_others"]:
    """Find the indices of the Others Pauli operators in the PauliOperators.

    Args:
        bits (UInt64[jnp.ndarray, "n_op 2n_packed"]):
            The bits of the Pauli operators.
        other_bits (UInt64[jnp.ndarray, "n_op_others 2n_packed"]):
            The bits of the Pauli operators to find.

    Returns:
        UInt64[jnp.ndarray, " n_op_others"]: The indices of the Others Pauli operators in the PauliOperators.

    """
    (n_op, n_packed) = bits.shape
    (n_op_others, _) = other_bits.shape

    def search_single(other: UInt64[jnp.ndarray, " 2n_packed"]) -> UInt64[jnp.ndarray, " 1"]:
        """Find the first index where `bits` is greater than or equal to `other`."""
        mask = jnp.all(bits >= other, axis=1)
        valid_rows = jnp.where(mask, jnp.arange(n_op), n_op)
        return jnp.min(valid_rows)

    return jax.vmap(search_single, in_axes=[0])(other_bits)


def new_sign(
    bits: UInt64[jnp.ndarray, "n_op n_packed"],
    other_bit: UInt64[jnp.ndarray, "1 n_packed"],
    signs: Complex128[jnp.ndarray, " n_op"],
    other_sign: Complex128[jnp.ndarray, " 1"],
) -> Complex128[jnp.ndarray, " n_op"]:
    """Update signs of Pauli operators during composition.

    Args:
        bits (UInt64[jnp.ndarray, "n_op 2n_packed"]):
            The bits of the first Pauli operator.
        other_bit (UInt64[jnp.ndarray, "1 2n_packed"]):
            The bits of the second Pauli operator.
        signs (Complex128[jnp.ndarray, "n_op"]):
            The signs of the first Pauli operator.
        other_sign (Complex128[jnp.ndarray, " 1"]):
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


def not_equal(
    bits: UInt64[jnp.ndarray, " n_op"],
    other_bits: UInt64[jnp.ndarray, " n_op"],
) -> Bool[jnp.ndarray, " n_op"]:
    result: Bool[jnp.ndarray, " n_op"] = bits != other_bits
    return result


def xor(
    bits: UInt64[jnp.ndarray, "n_op 2n_packed"],
    other_bit: UInt64[jnp.ndarray, "1 2n_packed"],
) -> UInt64[jnp.ndarray, " n_op 2n_packed"]:
    return jnp.bitwise_xor(bits, other_bit)


def pack_bits(bool_jnp: Bool[jnp.ndarray, "n_op n_qubits"]) -> UInt64[jnp.ndarray, "n_op n_packed"]:
    """Pack boolean array into uint64 array.

    Args:
        bool_jnp (Bool[jnp.ndarray, "n_op n_qubits"]):
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


def ztype_bool(
    bits: UInt64[jnp.ndarray, "n_op n_packed"],
) -> Bool[jnp.ndarray, " n_op"]:
    return jnp.logical_not(jnp.any(bits, axis=1))


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
