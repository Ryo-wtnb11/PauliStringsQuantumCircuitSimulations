from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax import export
from jaxtyping import Bool, UInt64

from paulistringsquantumcircuitsimulations.utils import (
    anticommutation,
    bits_equal,
    count_nonzero,
    find_bit_index,
    new_sign,
    not_equal,
    xor,
    ztype_bool,
)


def pack_bits_exported(
    n_qubits: int,
) -> Callable[[Bool[jnp.ndarray, "n_op n_qubits"]], UInt64[jnp.ndarray, "n_op n_packed"]]:
    """Create a compiled version of `pack_bits` for a fixed number of qubits.

    This function returns a callable that:
        - Takes a boolean JAX array of shape (n_op, n_qubits)
        - Returns a uint64 JAX array of shape (n_op, n_packed),
          where n_packed = ceil(n_qubits / 64)
        - n_op is symbolic (shape polymorphic), allowing reuse for different operator sizes

    The retur=ned function is compiled and avoids retracing on varying n_op.

    Args:
        n_qubits (int): The number of qubits per Pauli string
        (i.e. number of columns in the input boolean array)

    Returns:
        Callable: A compiled function with signature:
            (Bool[jnp.ndarray, "n_op n_qubits"]) -> UInt64[jnp.ndarray, "n_op n_packed"]

    Example:
        >>> pack_fn = pack_bits_exported(n_qubits=12)
        >>> x = jnp.array([[True, False, True, ...], ...], dtype=bool)  # shape = (n_op, 12)
        >>> packed = pack_fn(x)  # shape = (n_op, 1) # 1 = ceil(12 / 64)

    """
    n_packed = (n_qubits + 63) // 64

    @jax.jit
    def pack_bits(
        bool_jnp: Bool[jnp.ndarray, "n_op n_qubits"],
    ) -> UInt64[jnp.ndarray, "n_op n_packed"]:
        blocks = (jnp.arange(n_qubits) // 64).astype(jnp.uint64)
        positions = (jnp.arange(n_qubits) % 64).astype(jnp.uint64)
        bit_values = (bool_jnp * (1 << positions)).astype(jnp.uint64)

        def update_row(bit_row: UInt64[jnp.ndarray, "n_qubits"]) -> UInt64[jnp.ndarray, " n_packed"]:
            res_row: UInt64[jnp.ndarray, " n_packed"] = jnp.zeros(n_packed, dtype=jnp.uint64)
            return res_row.at[blocks].add(bit_row)

        return jax.vmap(update_row)(bit_values)

    # Shape polymorphic axis
    shapes = export.symbolic_shape(f"n_op, {n_qubits}")

    return export.export(pack_bits)(jax.ShapeDtypeStruct(shapes, jnp.bool_)).call


def find_bit_index_exported(n_qubits: int) -> Callable:
    two_n_packed = int(2 * ((n_qubits + 63) // 64))
    shapes = export.symbolic_shape(f"n_op, n_op_others, {two_n_packed}")

    return export.export(jax.jit(find_bit_index))(
        jax.ShapeDtypeStruct((shapes[0], shapes[2]), jnp.uint64),
        jax.ShapeDtypeStruct((shapes[1], shapes[2]), jnp.uint64),
    ).call


def bits_equal_exported(n_qubits: int) -> Callable:
    two_n_packed = int(2 * ((n_qubits + 63) // 64))
    shapes = export.symbolic_shape(f"n_op, {two_n_packed}")
    return export.export(jax.jit(bits_equal))(
        jax.ShapeDtypeStruct((shapes[0], shapes[1]), jnp.uint64),
        jax.ShapeDtypeStruct((shapes[0], shapes[1]), jnp.uint64),
    ).call


def count_nonzero_exported(n_qubits: int) -> Callable:
    n_packed = (n_qubits + 63) // 64
    shapes = export.symbolic_shape(f"n_op, {n_packed}")
    return export.export(jax.jit(count_nonzero))(
        jax.ShapeDtypeStruct(shapes, jnp.uint64),
    ).call


def new_sign_exported(n_qubits: int) -> Callable:
    n_packed = (n_qubits + 63) // 64
    shapes = export.symbolic_shape(f"n_op, 1, {n_packed}")
    return export.export(jax.jit(new_sign))(
        jax.ShapeDtypeStruct((shapes[0], shapes[2]), jnp.uint64),
        jax.ShapeDtypeStruct((shapes[1], shapes[2]), jnp.uint64),
        jax.ShapeDtypeStruct((shapes[0],), jnp.complex128),
        jax.ShapeDtypeStruct((shapes[1],), jnp.complex128),
    ).call


def xor_exported(n_qubits: int) -> Callable:
    two_n_packed = int(2 * ((n_qubits + 63) // 64))
    shapes = export.symbolic_shape(f"n_op, 1, {two_n_packed}")
    return export.export(jax.jit(xor))(
        jax.ShapeDtypeStruct((shapes[0], shapes[2]), jnp.uint64),
        jax.ShapeDtypeStruct((shapes[1], shapes[2]), jnp.uint64),
    ).call


def anticommutation_exported(n_qubits: int) -> Callable:
    n_packed = (n_qubits + 63) // 64
    shapes = export.symbolic_shape(f"n_op, 1, {n_packed}")
    return export.export(jax.jit(anticommutation))(
        jax.ShapeDtypeStruct((shapes[0], shapes[2]), jnp.uint64),
        jax.ShapeDtypeStruct((shapes[1], shapes[2]), jnp.uint64),
    ).call


def not_equal_exported() -> Callable:
    shapes = export.symbolic_shape("n_op")
    return export.export(jax.jit(not_equal))(
        jax.ShapeDtypeStruct(shapes, jnp.uint64),
        jax.ShapeDtypeStruct(shapes, jnp.uint64),
    ).call


def ztype_bool_exported(n_qubits: int) -> Callable:
    n_packed = (n_qubits + 63) // 64
    shapes = export.symbolic_shape(f"n_op, {n_packed}")
    return export.export(jax.jit(ztype_bool))(
        jax.ShapeDtypeStruct(shapes, jnp.uint64),
    ).call
