"""Quantum circuit simulation package for Pauli string evolution.

This package provides tools for simulating quantum circuits with a focus on
Pauli string evolution and expectation value calculations.
"""

import jax

jax.config.update("jax_enable_x64", True)  # noqa: FBT003
