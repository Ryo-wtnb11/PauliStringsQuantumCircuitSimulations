from typing import Self

import numpy as np
import numpy.typing as npt
import stim
from numba import njit, prange

from paulistringsquantumcircuitsimulations.exceptions import SystemSizeError

PauliString = str


@njit(parallel=True)  # type: ignore[misc]
def pack_bits(bool_array: npt.NDArray[np.bool_]) -> npt.NDArray[np.uint64]:
    (ndim_sum, ndim) = bool_array.shape
    ndim_out = (ndim + 63) // 64
    res = np.zeros((ndim_sum, ndim_out), dtype=np.uint64)

    blocks = np.arange(ndim) // 64
    positions = np.arange(ndim) % 64

    bit_values = (bool_array * (1 << positions)).astype(np.uint64)

    for i in prange(ndim_sum):
        for j in range(len(blocks)):
            res[i, blocks[j]] += bit_values[i, j]

    return res


@njit(parallel=True)  # type: ignore[misc]
def find_bit_index(bits: npt.NDArray[np.uint64], others: npt.NDArray[np.uint64]) -> npt.NDArray[np.int64]:
    (n_operators, nq) = bits.shape
    (n_operators_others, nq_others) = others.shape
    lower = np.repeat(0, n_operators_others)
    upper = np.repeat(n_operators, n_operators_others)
    for j in prange(n_operators_others):
        for i in range(nq_others):
            if upper[j] == lower[j]:
                break
            lower[j] = lower[j] + np.searchsorted(bits[lower[j] : upper[j], i], others[j, i], side="left")
            upper[j] = lower[j] + np.searchsorted(bits[lower[j] : upper[j], i], others[j, i], side="right")
    return lower


@njit(parallel=True)  # type: ignore[misc]
def bits_equal(bits: npt.NDArray[np.uint64], others: npt.NDArray[np.uint64]) -> npt.NDArray[np.bool_]:
    c = np.empty(len(others), dtype=np.bool_)
    for i in prange(len(c)):
        c[i] = np.all(bits[i, :] == others[i, :])
    return c


@njit(parallel=True)  # type: ignore[misc]
def insert_index(
    bits: np.ndarray,
    others: np.ndarray,
    signs: tuple[np.ndarray, np.ndarray],
    coefficients: tuple[np.ndarray, np.ndarray],
    index: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    (_, nq) = bits.shape
    new_size = len(bits) + len(others)
    bits_signs, others_signs = signs
    bits_coefficients, others_coefficients = coefficients
    res = np.empty((new_size, nq), dtype=np.uint64)
    res_signs = np.empty(new_size, dtype=np.complex128)
    res_coefficients = np.empty(new_size, dtype=np.complex128)

    insert_pos = index + np.arange(len(index))

    for i in prange(len(others)):
        res[insert_pos[i]] = others[i]
        res_signs[insert_pos[i]] = others_signs[i]
        res_coefficients[insert_pos[i]] = others_coefficients[i]

    bits_indices = np.zeros(new_size, dtype=np.int32)
    j = 0
    for i in range(new_size):
        if i not in insert_pos:
            bits_indices[j] = i
            j += 1

    for i in prange(len(bits)):
        res[bits_indices[i]] = bits[i]
        res_signs[bits_indices[i]] = bits_signs[i]
        res_coefficients[bits_indices[i]] = bits_coefficients[i]

    return res, res_signs, res_coefficients


@njit(parallel=True)  # type: ignore[misc]
def delete_index(
    bits: np.ndarray,
    signs: np.ndarray,
    coefficients: np.ndarray,
    index: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    (n_operators, nq) = bits.shape
    new_size = n_operators - len(index)
    res = np.empty((new_size, nq), dtype=np.uint64)
    res_s = np.empty(new_size, dtype=np.complex128)
    res_c = np.empty(new_size, dtype=np.complex128)
    mask = np.ones(n_operators, dtype=np.bool_)
    mask[index] = False
    ind = np.nonzero(mask)[0]
    for i in prange(len(ind)):
        res[i] = bits[ind[i]]
        res_s[i] = signs[ind[i]]
        res_c[i] = coefficients[ind[i]]
    return res, res_s, res_c


@njit  # type: ignore[misc]
def count_set_bits(n: int) -> int:
    count = 0
    while n:
        count += n & 1
        n >>= 1
    return count


@njit  # type: ignore[misc]
def count_nonzero(bits: npt.NDArray[np.uint64]) -> int:
    s = 0
    for i in range(len(bits)):
        s += count_set_bits(bits[i])
    return s


@njit(parallel=True)  # type: ignore[misc]
def anticommutation(bits1: npt.NDArray[np.uint64], bits2: npt.NDArray[np.uint64]) -> npt.NDArray[np.bool_]:
    res = np.empty(len(bits1), dtype=np.int64)
    for i in prange(len(bits1)):
        res[i] = count_nonzero(np.bitwise_and(bits1[i, :], bits2[:]))
    return np.mod(res, 2)


@njit(parallel=True)  # type: ignore[misc]
def update_sign(
    sign1: npt.NDArray[np.complex128],
    sign2: npt.NDArray[np.complex128],
    bits1: npt.NDArray[np.uint64],
    bits2: npt.NDArray[np.uint64],
) -> None:
    """Update signs of Pauli operators during composition.

    Args:
        sign1: Array of signs {1, -1, 1j, -1j}
        sign2: Single sign {1, -1, 1j, -1j}
        bits1: First operator's bits
        bits2: Second operator's bits

    """
    for i in prange(len(sign1)):
        n_common = np.count_nonzero(np.bitwise_and(bits1[i, :], bits2[:]))
        sign1[i] = sign1[i] * sign2 * ((-1j) ** (2 * n_common))


@njit(parallel=True)  # type: ignore[misc]
def not_equal(bits: npt.NDArray[np.uint64], bits_others: npt.NDArray[np.uint64]) -> npt.NDArray[np.bool_]:
    c = np.empty(len(bits), dtype=np.bool_)
    c = bits != bits_others
    return c.astype(np.bool_)


@njit(parallel=True)  # type: ignore[misc]
def inplace_xor(bits1: npt.NDArray[np.uint64], bits2: npt.NDArray[np.uint64]) -> None:
    bits1[:, :] = np.bitwise_xor(bits1, bits2)


class PauliOperators:
    """A class for representing a list of Pauli operators.

    Attributes:
        bits: npt.NDArray[np.uint64]
            The bits of the Pauli operators.
        signs: npt.NDArray[np.complex128]
            The signs of the Pauli operators.
        coefficients: npt.NDArray[np.complex128]
            The coefficients of the Pauli operators.
        nq: int
            The number of qubits.

    """

    def __init__(
        self,
        bits: npt.NDArray[np.uint64],
        signs: npt.NDArray[np.complex128],
        coefficients: npt.NDArray[np.complex128],
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

        signs_ = np.array(signs, dtype=np.complex128)
        coefficients_ = np.array(coefficients, dtype=np.complex128)

        paulis: list[stim.PauliString] = [stim.PauliString(ps) for ps in paulistrings]
        xs, zs = zip(*[ps.to_numpy() for ps in paulis], strict=False)
        xs_ = np.array(xs, dtype=np.bool_)
        zs_ = np.array(zs, dtype=np.bool_)
        bits = np.hstack((pack_bits(xs_), pack_bits(zs_)))

        for ps in paulistrings:
            if len(ps) != n_qubits:
                raise SystemSizeError(len(ps), n_qubits)

        return cls(bits, signs_, coefficients_, n_qubits)

    def order_paulis(self) -> None:
        (n_operators, nq_orders) = self.size()
        indices = np.lexsort([self.bits[:, j] for j in reversed(range(nq_orders))])
        self.bits = self.bits[indices]
        self.signs = self.signs[indices]
        self.coefficients = self.coefficients[indices]

    def find_pauli_indices(self, others: Self) -> npt.NDArray[np.int64]:
        """Find the indices of the Others Pauli operators in the PauliOperators.

        Args:
            others: Self
                The PauliOperators to find the indices of.

        Returns:
            npt.NDArray[np.int64]: The indices of the Others Pauli operators in the PauliOperators.

        """
        result: npt.NDArray[np.int64] = find_bit_index(self.bits, others.bits)
        return result

    def find_pauli(self, others: Self, index: npt.NDArray[np.int64] | None = None) -> npt.NDArray[np.bool_]:
        """Find the indices of the Others Pauli operators in the PauliOperators.

        Args:
            others: Self
                The PauliOperators to find the indices of.
            index: npt.NDArray[np.int64] | None
                The indices of the Pauli operators to find.

        Returns:
            npt.NDArray[np.bool_]: The indices of the Others Pauli operators in the PauliOperators.

        """
        if index is None:
            index = self.find_pauli_indices(others)
        (n_operators, _) = self.size()
        result: npt.NDArray[np.bool_] = bits_equal(self.bits[index % n_operators, :], others.bits)
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

    def delete_pauli(self, index: npt.NDArray[np.int64]) -> None:
        """Delete the Pauli operators at the given indices.

        Args:
            index: npt.NDArray[np.int64]
                The indices of the Pauli operators to delete.

        """
        self.bits, self.signs, self.coefficients = delete_index(
            self.bits,
            self.signs,
            self.coefficients,
            index,
        )

    def anticommutes(self, other: Self) -> npt.NDArray[np.bool_]:
        """Check if the Pauli operators anticommute.

        Args:
            other: Self
                The PauliOperators to check if they anticommute.

        Returns:
            npt.NDArray[np.bool_]: Array indicating which operators anticommute.

        """
        nq = (self.n_qubits + 63) // 64

        a_dot_b = anticommutation(self.bits[:, nq:], other.bits[0, :nq])
        b_dot_a = anticommutation(self.bits[:, :nq], other.bits[0, nq:])
        result: npt.NDArray[np.bool_] = not_equal(a_dot_b, b_dot_a)
        return result

    def compose_with(self, other: Self) -> None:
        """Composes all Paulis in 'self' with the Pauli (only one Pauli allowed) in 'other'.

        Args:
            other: Self
                The PauliOperators to compose with.

        """
        nq = (self.n_qubits + 63) // 64
        update_sign(
            self.signs[:],
            other.signs[0],
            self.bits[:, :nq],
            other.bits[0, nq:],
        )
        inplace_xor(self.bits, other.bits[0, :])

    def ztype(self, index: npt.NDArray[np.int64] | None = None) -> npt.NDArray[np.bool_]:
        """Return logical array indicating whether a Pauli in self is composed only of Z or identity Pauli.

        Args:
            index: npt.NDArray[np.int64] | None
                The indices of the Pauli operators to check, if needed.

        Returns:
            npt.NDArray[np.bool_]: Array indicating which operators are Z or identity.

        """
        nq = (self.n_qubits + 63) // 64
        if index is None:
            result: npt.NDArray[np.bool_] = np.logical_not(np.any(self.bits[:, nq:], axis=1))
        else:
            result = np.logical_not(np.any(self.bits[index, nq:], axis=1))
        return result

    def size(self) -> tuple[int, ...]:
        """Get the size of the bits array.

        Returns:
            The size of the bits array.

        """
        return self.bits.shape
