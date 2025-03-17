"""Tests for Observable module."""

import jax.numpy as jnp
import pytest
import stim

from paulistringsquantumcircuitsimulations.observable import Observable


def test_observable_creation() -> None:
    """Test Observable instance creation."""
    obs = Observable(coefficient=1.0, paulistring="Z")
    assert obs.coefficient == 1.0
    assert str(obs.paulistring) == "Z"

def test_expectation() -> None:
    """Test expectation value."""
    obs = Observable(coefficient=1.0, paulistring="Z")
    assert obs.expectation("0") == 1.0
    assert obs.expectation("1") == -1.0

    obs = Observable(coefficient=1.0, paulistring="XZ")
    assert obs.expectation("00") == 0.0
    assert obs.expectation("01") == 0.0
    assert obs.expectation("10") == 0.0
    assert obs.expectation("11") == 0.0

    obs = Observable(coefficient=1.0, paulistring="_Z")
    assert obs.expectation("00") == 1.0
    assert obs.expectation("01") == -1.0
    assert obs.expectation("10") == 1.0
    assert obs.expectation("11") == -1.0
