from typing import Self

import jax.numpy as jnp
import stim

PauliString = str


class Observable:
    """Observable.

    Args:
        value (jnp.ndarray): The value of the observable.
        pauli_string (PauliString): The PauliString representation of the observable.

    """

    def __init__(self, coefficient: jnp.complex128 | jnp.float64, paulistring: PauliString) -> None:
        self.coefficient: jnp.complex128 = jnp.complex128(coefficient)
        self.paulistring: PauliString = paulistring

    def commutes(self, other: Self | PauliString) -> bool:
        """Check if this observable commutes with another.

        Args:
            other: Another Observable instance to check commutation with

        Returns:
            bool: True if the observables commute, False otherwise

        """
        return bool(
            stim.PauliString(self.paulistring).commutes(
                stim.PauliString(other.paulistring if isinstance(other, type(self)) else other),
            ),
        )

    def expectation(self, state: str = "") -> jnp.ndarray:
        """Calculate expectation value of observables for given computational basis state.

        Args:
            state (str): Computational basis state (e.g., "111000"). If empty, uses |0...0⟩ state.

        Returns:
            jnp.ndarray: Expectation value. Can be differentiated using JAX.

        """
        paulistring = stim.PauliString(self.paulistring)
        xs, zs = paulistring.to_numpy()
        xs = jnp.array(xs, dtype=jnp.bool_)
        zs = jnp.array(zs, dtype=jnp.bool_)

        # If state is empty, use |0...0⟩ state
        if not state:
            state = "0" * len(xs)
        # Convert state to boolean array: "0" -> False, "1" -> True
        states = jnp.array([int(bit) for bit in state], dtype=jnp.bool_)

        # If any X operator acts on a 1-qubit, expectation is zero
        if jnp.any(xs):
            return jnp.array(0.0, dtype=jnp.float64)

        # Compute expectation from Z operators
        z_contributions = jnp.where(states, -1.0, 1.0)  # Z|1> = -|1>, Z|0> = |0>
        expectation_value = jnp.prod(jnp.where(zs, z_contributions, 1))
        return jnp.array(jnp.real(self.coefficient * expectation_value * paulistring.sign), dtype=jnp.float64)
