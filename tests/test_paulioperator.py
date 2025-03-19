import numpy as np
import stim

from paulistringsquantumcircuitsimulations.paulioperators import PauliOperators


def test_paulioperator_init() -> None:
    paulistrings = ["___", "X__", "Y__", "Z__", "XX_", "XY_"]
    pauli_operators = PauliOperators.from_strings(paulistrings=paulistrings, n_qubits=3)
    pauli_operators.order_paulis()
    assert np.all(pauli_operators.bits == np.array([
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
            [3, 0],
            [3, 2],
        ],
        dtype=np.uint64,
    ))


    others = ["ZX_", "XX_", "YYY"]
    other_pauli_operators = PauliOperators.from_strings(paulistrings=others, n_qubits=3)
    assert np.all(pauli_operators.find_pauli_indices(other_pauli_operators) == np.array([4, 4, 6]))
    assert np.all(pauli_operators.find_pauli(other_pauli_operators) == np.array([False, True, False]))


