from typing import Self

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
    others: UInt64[jnp.ndarray, "n_op_others n_packed"],
) -> UInt64[jnp.ndarray, " n_op_others"]:
    """Find the indices of the Others Pauli operators in the PauliOperators.

    Args:
        bits: UInt64[jnp.ndarray, "n_op n_packed"]
            The bits of the Pauli operators.
        others: UInt64[jnp.ndarray, "n_op_others n_packed"]
            The bits of the Pauli operators to find.

    Returns:
        UInt64[jnp.ndarray, " n_op_others"]: The indices of the Others Pauli operators in the PauliOperators.

    """
    (n_op, n_packed) = bits.shape
    (n_op_others, _) = others.shape

    def search_single(other: UInt64[jnp.ndarray, " n_packed"]) -> UInt64[jnp.ndarray, " 1"]:
        """Find the first index where `bits` is greater than or equal to `other`."""
        mask = jnp.all(bits >= other, axis=1)
        valid_rows = jnp.where(mask, jnp.arange(n_op), n_op)
        return jnp.min(valid_rows)

    return jax.vmap(search_single, in_axes=[0])(others)


@jax.jit
def bits_equal(
    bits: UInt64[jnp.ndarray, "n_op n_packed"],
    others: UInt64[jnp.ndarray, "n_op_others n_packed"],
) -> Bool[jnp.ndarray, " n_op_others"]:
    """Compare bits arrays element-wise.

    Args:
        bits: UInt64[jnp.ndarray, "n_op n_packed"]
            First bits array.
        others: UInt64[jnp.ndarray, "n_op_others n_packed"]
            Second bits array.

    Returns:
        Bool[jnp.ndarray, "n_op_others"]: Boolean array indicating where bits are equal.

    """
    return jnp.all(bits == others, axis=1)


def insert_index(
    bits: UInt64[jnp.ndarray, "n_op n_packed"],
    others: UInt64[jnp.ndarray, "n_op_others n_packed"],
    signs: tuple[Complex128[jnp.ndarray, " n_op"], Complex128[jnp.ndarray, " n_op_others"]],
    coefficients: tuple[Complex128[jnp.ndarray, " n_op"], Complex128[jnp.ndarray, " n_op_others"]],
    index: UInt64[jnp.ndarray, " n_op_others"],
) -> tuple[
    UInt64[jnp.ndarray, " n_op_new n_packed"],
    Complex128[jnp.ndarray, " n_op_new"],
    Complex128[jnp.ndarray, " n_op_new"],
]:
    """Insert `others` into `bits` at positions `index`, updating `signs` and `coefficients` accordingly.

    Args:
        bits: UInt64[jnp.ndarray, "n_op n_packed"]
            The bits of the Pauli operators.
        others: UInt64[jnp.ndarray, "n_op_others n_packed"]
            The bits of the Pauli operators to insert.
        signs: tuple[Complex128[jnp.ndarray, " n_op"], Complex128[jnp.ndarray, " n_op_others"]]
            The signs of the Pauli operators.
        coefficients: tuple[Complex128[jnp.ndarray, " n_op"], Complex128[jnp.ndarray, " n_op_others"]]
            The coefficients of the Pauli operators.
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
    n_op_others = len(others)
    new_size = n_op + n_op_others

    bits_signs, others_signs = signs
    bits_coefficients, others_coefficients = coefficients

    res_bits = jnp.zeros((new_size, n_packed), dtype=jnp.uint64)
    res_signs = jnp.zeros(new_size, dtype=jnp.complex128)
    res_coefficients = jnp.zeros(new_size, dtype=jnp.complex128)

    res_bits = jnp.zeros((new_size, n_packed), dtype=jnp.uint64)
    res_signs = jnp.zeros(new_size, dtype=jnp.complex128)
    res_coefficients = jnp.zeros(new_size, dtype=jnp.complex128)

    insert_pos = index + jnp.arange(n_op_others)

    res_bits = res_bits.at[insert_pos].set(others)
    res_signs = res_signs.at[insert_pos].set(others_signs)
    res_coefficients = res_coefficients.at[insert_pos].set(others_coefficients)

    all_indices = jnp.arange(new_size)
    mask = ~jnp.isin(all_indices, insert_pos)
    bits_indices = all_indices[mask]

    res_bits = res_bits.at[bits_indices].set(bits)
    res_signs = res_signs.at[bits_indices].set(bits_signs)
    res_coefficients = res_coefficients.at[bits_indices].set(bits_coefficients)

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
    bits1: UInt64[jnp.ndarray, "n_op n_packed"],
    bits2: UInt64[jnp.ndarray, "1 n_packed"],
) -> UInt64[jnp.ndarray, " n_op"]:
    return jax.vmap(lambda row: jnp.mod(count_nonzero(jnp.bitwise_and(row, bits2[0])), 2))(bits1)


@jax.jit
def new_sign(
    sign1: Complex128[jnp.ndarray, " n_op"],
    sign2: Complex128[jnp.ndarray, " 1"],
    bits1: UInt64[jnp.ndarray, "n_op n_packed"],
    bits2: UInt64[jnp.ndarray, "1 n_packed"],
) -> Complex128[jnp.ndarray, " n_op"]:
    """Update signs of Pauli operators during composition.

    Args:
        sign1: Complex128[jnp.ndarray, "n_op"]
            The signs of the first Pauli operator.
        sign2: Complex128[jnp.ndarray, " 1"]
            The sign of the second Pauli operator.
        bits1: UInt64[jnp.ndarray, "n_op n_packed"]
            The bits of the first Pauli operator.
        bits2: UInt64[jnp.ndarray, "1 n_packed"]
            The bits of the second Pauli operator.

    """

    def compute_sign(
        row: UInt64[jnp.ndarray, " n_packed"],
        s1: Complex128[jnp.ndarray, " 1"],
    ) -> Complex128[jnp.ndarray, " 1"]:
        n_common = jnp.count_nonzero(jnp.bitwise_and(row, bits2[0]))
        return s1 * sign2 * ((-1j) ** (2 * n_common))

    return jax.vmap(compute_sign, in_axes=[0, 0])(bits1, sign1)


def not_equal(
    bits1: UInt64[jnp.ndarray, " n_op"],
    bits2: UInt64[jnp.ndarray, " n_op"],
) -> Bool[jnp.ndarray, " n_op"]:
    return bits1 != bits2


@jax.jit
def xor(
    bits1: UInt64[jnp.ndarray, "n_op n_packed"],
    bits2: UInt64[jnp.ndarray, "1 n_packed"],
) -> UInt64[jnp.ndarray, " n_op n_packed"]:
    return jnp.bitwise_xor(bits1, bits2)


class PauliOperators:
    """A class for representing a list of Pauli operators.

    Attributes:
        bits: UInt64[jnp.ndarray, "n_op n_packed"]
            The bits of the Pauli operators.
        signs: Complex128[jnp.ndarray, " n_op"]
            The signs of the Pauli operators.
        coefficients: Complex128[jnp.ndarray, " n_op"]
            The coefficients of the Pauli operators.
        nq: int
            The number of qubits.

    """

    def __init__(
        self,
        bits: UInt64[jnp.ndarray, "n_op n_packed"],
        signs: Complex128[jnp.ndarray, " n_op"],
        coefficients: Complex128[jnp.ndarray, " n_op"],
        n_qubits: int,
    ) -> None:
        self.bits = bits
        self.signs = signs
        self.coefficients = coefficients
        self.n_qubits = n_qubits

    @classmethod
    def from_strings(
        cls,
        paulistrings: list[PauliString],
        n_qubits: int,
        signs: list[complex] | None = None,
        coefficients: list[complex] | None = None,
    ) -> Self:
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

        """
        if signs is None:
            signs = [1.0 + 0.0j] * len(paulistrings)
        if coefficients is None:
            coefficients = [1.0 + 0.0j] * len(paulistrings)

        signs_ = jnp.array(signs, dtype=jnp.complex128)
        coefficients_ = jnp.array(coefficients, dtype=jnp.complex128)

        paulis: list[stim.PauliString] = [stim.PauliString(ps) for ps in paulistrings]
        xs, zs = zip(*[ps.to_numpy() for ps in paulis], strict=False)
        xs_ = jnp.array(xs, dtype=jnp.bool_)
        zs_ = jnp.array(zs, dtype=jnp.bool_)
        bits = jnp.hstack((pack_bits(zs_), pack_bits(xs_)))

        for ps in paulistrings:
            if len(ps) != n_qubits:
                raise SystemSizeError(len(ps), n_qubits)

        return cls(bits, signs_, coefficients_, n_qubits)

    def order_paulis(self) -> None:
        (n_operators, nq_orders) = self.size()
        indices = jnp.lexsort([self.bits[:, j] for j in reversed(range(nq_orders))])
        self.bits = self.bits[indices]
        self.signs = self.signs[indices]
        self.coefficients = self.coefficients[indices]

    def find_pauli_indices(self, others: Self) -> UInt64[jnp.ndarray, " n_op_others"]:
        """Find the indices of the Others Pauli operators in the PauliOperators.

        Args:
            others: Self
                The PauliOperators to find the indices of.

        Returns:
            UInt64[jnp.ndarray, " n_op_others"]:
            The indices of the Others Pauli operators in the PauliOperators.

        """
        result: UInt64[jnp.ndarray, " n_op_others"] = find_bit_index(self.bits, others.bits)
        return result

    def find_pauli(
        self,
        others: Self,
        index: UInt64[jnp.ndarray, " n_op_others"] | None = None,
    ) -> UInt64[jnp.ndarray, " n_op_others"]:
        """Find the indices of the Others Pauli operators in the PauliOperators.

        Args:
            others: Self
                The PauliOperators to find the indices of.
            index: UInt64[jnp.ndarray, " n_op_others"] | None
                The indices of the Pauli operators to find.

        Returns:
            UInt64[jnp.ndarray, " n_op_others"]:
            The indices of the Others Pauli operators in the PauliOperators.

        """
        if index is None:
            index = self.find_pauli_indices(others)
        (n_operators, _) = self.size()
        result: UInt64[jnp.ndarray, " n_op_others"] = bits_equal(
            self.bits[index % n_operators, :],
            others.bits,
        )
        return result

    def insert_pauli(
        self,
        others: Self,
    ) -> None:
        """Insert a new Pauli or a list of Paulis (stored in PauliRepresentation 'other') into 'self'.

        Args:
            others: Self
                The PauliOperators to insert.

        """
        index = self.find_pauli_indices(others)
        self.bits, self.signs, self.coefficients = insert_index(
            self.bits,
            others.bits,
            (self.signs, others.signs),
            (self.coefficients, others.coefficients),
            index,
        )

    def delete_pauli(self, index: UInt64[jnp.ndarray, " n_op_others"]) -> None:
        """Delete the Pauli operators at the given indices.

        Args:
            index: UInt64[jnp.ndarray, " n_op_others"]
                The indices of the Pauli operators to delete.

        """
        self.bits, self.signs, self.coefficients = delete_index(
            self.bits,
            self.signs,
            self.coefficients,
            index,
        )

    def anticommutes(self, other: Self) -> Bool[jnp.ndarray, " n_op"]:
        """Check if the Pauli operators anticommute.

        Args:
            other: Self
                The PauliOperators to check if they anticommute.

        Returns:
            Bool[jnp.ndarray, " n_op"]: Array indicating which operators anticommute.

        """
        nq = (self.n_qubits + 63) // 64
        a_dot_b = anticommutation(self.bits[:, nq:], other.bits[0, :nq])
        b_dot_a = anticommutation(self.bits[:, :nq], other.bits[0, nq:])
        result: Bool[jnp.ndarray, " n_op"] = not_equal(a_dot_b, b_dot_a)
        return result

    def compose_with(self, other: Self) -> None:
        """Composes all Paulis in 'self' with the Pauli (only one Pauli allowed) in 'other'.

        Args:
            other: Self
                The PauliOperators to compose with.

        """
        nq = (self.n_qubits + 63) // 64
        self.signs = new_sign(
            self.signs[:],
            other.signs[0],
            self.bits[:, :nq],
            other.bits[0, nq:],
        )
        self.bits = xor(self.bits, other.bits[0, :])

    def ztype(
        self,
        index: UInt64[jnp.ndarray, " n_op_others"] | None = None,
    ) -> Bool[jnp.ndarray, " n_op_others"]:
        """Return logical array indicating whether a Pauli in self is composed only of Z or identity Pauli.

        Args:
            index: npt.NDArray[np.int64] | None
                The indices of the Pauli operators to check, if needed.

        Returns:
            npt.NDArray[np.bool_]: Array indicating which operators are Z or identity.

        """
        nq = (self.n_qubits + 63) // 64
        if index is None:
            result: Bool[jnp.ndarray, " n_op"] = jnp.logical_not(jnp.any(self.bits[:, nq:], axis=1))
        else:
            result = jnp.logical_not(jnp.any(self.bits[index, nq:], axis=1))
        return result

    def size(self) -> tuple[int, ...]:
        """Get the size of the bits array.

        Returns:
            The size of the bits array.

        """
        return self.bits.shape
