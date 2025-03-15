"""Quantum circuit simulation package for Pauli string evolution.

This package provides tools for simulating quantum circuits with a focus on
Pauli string evolution and expectation value calculations.
"""

import jax

from .circuit import Gate
from .observable import Observable, expectation, operator_evolution

jax.config.update("jax_enable_x64", True)  # noqa: FBT003
__all__ = ["Gate", "Observable", "expectation", "operator_evolution"]
