import numpy as np
import stim

from paulistringsquantumcircuitsimulations.paulioperator import PauliOperator


def test_paulioperator_init() -> None:
    pauli_operator = PauliOperator.from_string("ZXZ")
    assert pauli_operator.sign == 1
    assert np.all(pauli_operator.xs == stim.PauliString("ZXZ").to_numpy(bit_packed=True)[0])
    assert np.all(pauli_operator.zs == stim.PauliString("ZXZ").to_numpy(bit_packed=True)[1])

    pauli_operator = PauliOperator.from_string("-ZXZ")
    assert pauli_operator.sign == -1

    pauli_operator = PauliOperator.from_string("-iZXZ")
    assert pauli_operator.sign == -1j