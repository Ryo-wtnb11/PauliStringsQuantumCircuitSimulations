import jax.numpy as jnp
import stim

from paulistringsquantumcircuitsimulations.paulioperators import (
    paulioperators_from_strings,
    order_paulioperators,
    find_paulioperators_indices,
    find_paulioperators,
)


def test_paulioperator_init() -> None:
    paulistrings = ["___", "X__", "Y__", "Z__", "XX_", "XY_"]
    bits, signs, coefficients = paulioperators_from_strings(paulistrings=paulistrings, n_qubits=3)
    bits, signs, coefficients = order_paulioperators(bits, signs, coefficients)
    assert jnp.all(bits == jnp.array([
            [0, 0],
            [0, 1],
            [0, 3],
            [1, 0],
            [1, 1],
            [2, 3],
        ],
        dtype=jnp.uint64,
    ))


    others = ["ZX_", "XX_", "YYY"]
    other_bits, other_signs, other_coefficients = paulioperators_from_strings(paulistrings=others, n_qubits=3)
    assert jnp.all(find_paulioperators_indices(bits, other_bits) == jnp.array([5, 2, 6]))
    assert jnp.all(find_paulioperators(bits, other_bits) == jnp.array([False, True, False]))


