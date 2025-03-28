import numpy as np
from jaxtyping import Bool, Int32, UInt32
from numba import njit, prange


@njit(fastmath=True)  # type: ignore [misc]
def pack_bits(
    bits: Bool[np.ndarray, " n_operators n_bits"],
) -> UInt32[np.ndarray, " n_operators n_packed"]:
    n_operators, n_bits = bits.shape
    packed_bits: UInt32[np.ndarray, " n_operators n_packed"] = np.zeros(
        (n_operators, (n_bits + 31) // 32), dtype=np.uint32
    )
    for i in range(n_operators):
        for j in range(n_bits):
            packed_bits[i, j // 32] |= bits[i, j] << (j % 32)
    return packed_bits


@njit(parallel=True, fastmath=True)  # type: ignore [misc]
def update_phase(
    bits: UInt32[np.ndarray, " n_operators n_packed"],
    other_bit: UInt32[np.ndarray, " 1 n_packed"],
    phases: Int32[np.ndarray, " n_operators"],
    other_phase: Int32[np.ndarray, " 1"],
) -> Int32[np.ndarray, " n_operators"]:
    n_operators = bits.shape[0]
    res = np.zeros(n_operators)
    for i in prange(n_operators):
        res[i] = phases[i] + other_phase[0] + 2 * count_nonzero(np.bitwise_and(bits[i], other_bit[0]))
    return res


@njit(parallel=True, fastmath=True)  # type: ignore [misc]
def insert_index_bits_and_phases(
    bits: UInt32[np.ndarray, " n_operators n_packed"],
    other_bits: UInt32[np.ndarray, " n_other_operators n_packed"],
    phases: Int32[np.ndarray, " n_operators"],
    other_phases: Int32[np.ndarray, " n_other_operators"],
    index: UInt32[np.ndarray, " n_other_operators"],
) -> tuple[UInt32[np.ndarray, " new_n_operators n_packed"], Int32[np.ndarray, " new_n_operators"]]:
    n_operators, two_n_packed = bits.shape
    n_other_operators, _ = other_bits.shape
    new_size = n_operators + n_other_operators
    res = np.empty((new_size, two_n_packed), dtype=np.uint32)
    res_p = np.empty(new_size, dtype=np.int32)
    ind = index + np.arange(len(index))
    res[: ind[0]] = bits[: index[0]]
    res_p[: ind[0]] = phases[: index[0]]
    for i in prange(len(ind)):
        res[ind[i]] = other_bits[i]
        res_p[ind[i]] = other_phases[i]
        if i == len(ind) - 1:
            u = new_size
            ua = n_operators
        else:
            u = ind[i + 1]
            ua = index[i + 1]
        res[ind[i] + 1 : u] = bits[index[i] : ua]
        res_p[ind[i] + 1 : u] = phases[index[i] : ua]
    return res, res_p


@njit(parallel=True, fastmath=True)  # type: ignore [misc]
def delete_index_bits_and_phases(
    bits: UInt32[np.ndarray, " n_operators n_packed"],
    phases: Int32[np.ndarray, " n_operators"],
    index: UInt32[np.ndarray, " n_other_operators"],
) -> tuple[UInt32[np.ndarray, " new_n_operators n_packed"], Int32[np.ndarray, " new_n_operators"]]:
    n_operators, two_n_packed = bits.shape
    new_size = n_operators - len(index)
    res = np.empty((new_size, two_n_packed), dtype=np.uint32)
    res_p = np.empty(new_size, dtype=np.int32)
    mask = np.ones(n_operators, dtype=np.bool_)
    mask[index] = False
    ind = np.nonzero(mask)[0]
    for i in prange(len(ind)):
        res[i] = bits[ind[i]]
        res_p[i] = phases[ind[i]]
    return res, res_p


@njit(parallel=True, fastmath=True)  # type: ignore [misc]
def count_nonzero(
    bits_and_other_bits: Int32[np.ndarray, " n_packed"],
) -> int:
    s: int = 0
    for i in range(len(bits_and_other_bits)):
        s += count_set_bits(bits_and_other_bits[i])
    return s


@njit(fastmath=True)  # type: ignore [misc]
def count_set_bits(x: int) -> int:
    x = x - ((x >> 1) & 0x5555555555555555)
    x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333)
    x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0F
    x = x + (x >> 8)
    x = x + (x >> 16)
    x = x + (x >> 32)
    return int(x & 0x7F)


@njit(fastmath=True)  # type: ignore [misc]
def xor(
    bits: Int32[np.ndarray, " n_operators n_packed"],
    other_bit: Int32[np.ndarray, " 1 n_packed"],
) -> Int32[np.ndarray, " n_operators n_packed"]:
    return np.bitwise_xor(bits, other_bit[0, :])


@njit(parallel=True, fastmath=True)  # type: ignore [misc]
def anticommutation(
    bits: UInt32[np.ndarray, " n_operators n_packed"],
    other_bit: UInt32[np.ndarray, " 1 n_packed"],
) -> UInt32[np.ndarray, " n_operators"]:
    n_operators, n_packed = bits.shape
    res = np.empty(n_operators, dtype=np.int16)
    for i in prange(n_operators):
        res[i] = np.mod(count_nonzero(np.bitwise_and(bits[i, :], other_bit[0, :])), 2)
    return res


@njit  # type: ignore [misc]
def not_equal(
    a: UInt32[np.ndarray, " n_operators"],
    b: UInt32[np.ndarray, " n_operators"],
) -> Bool[np.ndarray, " n_operators"]:
    res: Bool[np.ndarray, " n_operators"] = a != b
    return res


@njit(parallel=True, fastmath=True)  # type: ignore [misc]
def bits_equal(
    bits: UInt32[np.ndarray, " n_operators n_packed"],
    other_bits: UInt32[np.ndarray, " n_other_operators n_packed"],
) -> Bool[np.ndarray, " n_other_operators"]:
    n_other_operators, _ = other_bits.shape
    res = np.empty(n_other_operators, dtype=np.bool_)
    for i in prange(n_other_operators):
        res[i] = np.all(bits[i, :] == other_bits[i, :])
    return res


@njit(parallel=True, fastmath=True)  # type: ignore [misc]
def bits_equal_index(
    bits: UInt32[np.ndarray, " n_operators n_packed"],
    other_bits: UInt32[np.ndarray, " n_other_operators n_packed"],
    index: UInt32[np.ndarray, " n_other_operators"],
) -> Bool[np.ndarray, " n_other_operators"]:
    n_other_operators, n_packed = other_bits.shape
    res = np.empty(n_other_operators, dtype=np.bool_)
    for i in prange(n_other_operators):
        res[i] = ~np.any(bits[index[i], :] != other_bits[i, :])
    return res


@njit(parallel=True, fastmath=True)  # type: ignore [misc]
def find_bit_index(
    bits: UInt32[np.ndarray, " n_operators 2n_packed"],
    other_bits: UInt32[np.ndarray, " n_other_operators 2n_packed"],
) -> UInt32[np.ndarray, " n_other_operators"]:
    n_operators, _ = bits.shape
    n_other_operators, two_n_packed = other_bits.shape
    lower = np.repeat(0, n_other_operators)
    upper = np.repeat(n_operators, n_other_operators)
    for j in prange(n_other_operators):
        for i in range(two_n_packed):
            if upper[j] == lower[j]:
                break
            lower[j] = lower[j] + np.searchsorted(bits[lower[j] : upper[j], i], other_bits[j, i], side="left")
            upper[j] = lower[j] + np.searchsorted(
                bits[lower[j] : upper[j], i], other_bits[j, i], side="right"
            )
    return lower
