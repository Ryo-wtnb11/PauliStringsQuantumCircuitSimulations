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

    def __init__(self, coefficient: jnp.float64, paulistring: PauliString) -> None:
        self.coefficient: jnp.float64 = coefficient
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

    def expectation(self) -> jnp.ndarray:
        """Calculate expectation value of observables for |0...0⟩ state.

        Returns:
            jnp.ndarray: Expectation value. Can be differentiated using JAX.

        Example:
            >>> obs = Observable(coefficient=1.0, paulistring=stim.PauliString("Z"))
            >>> obs.expectation()  # ⟨0|Z|0⟩ = 1.0
            DeviceArray(1., dtype=float64)

        """
        paulistring = stim.PauliString(self.paulistring)
        xs, zs = paulistring.to_numpy()
        xs = jnp.array(xs, dtype=jnp.bool)
        zs = jnp.array(zs, dtype=jnp.bool)

        if jnp.any(xs):
            return jnp.array(0.0, dtype=jnp.float64)
        return jnp.array(jnp.real(self.coefficient * paulistring.sign), dtype=jnp.float64)
