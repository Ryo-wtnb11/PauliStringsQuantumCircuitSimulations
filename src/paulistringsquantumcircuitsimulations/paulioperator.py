from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import stim

PauliString = str
BoolOrIntArray = npt.NDArray[np.bool_] | npt.NDArray[np.int64]


@dataclass
class PauliOperator:
    sign: np.complex128
    xs: npt.NDArray[np.uint8]
    zs: npt.NDArray[np.uint8]

    @classmethod
    def from_string(cls, paulistring: PauliString) -> "PauliOperator":
        ps = stim.PauliString(paulistring)
        sign = ps.sign
        xs, zs = ps.to_numpy(bit_packed=True)
        return cls(sign, xs, zs)
