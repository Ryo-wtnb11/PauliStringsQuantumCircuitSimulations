class CircuitSystemSizeError(ValueError):
    """Raised when gate targets exceed system size."""

    def __init__(self, target: int, n: int) -> None:
        super().__init__(f"Qubit index {target} exceeds system size {n}")


class SystemSizeError(ValueError):
    """Raised when Pauli operator length exceeds expected size."""

    def __init__(self, n1: int, n2: int, *, packed: bool = False) -> None:
        if packed:
            super().__init__(
                f"Mismatch between system size {n1} and {n2}. \n"
                "Note that pauli operators are packed by uint64"
                "so the actual length is close to be 64 times the number of qubits",
            )
        else:
            super().__init__(
                f"Mismatch between system size {n1} and {n2}.",
            )


class InvalidGateError(ValueError):
    """Raised when an invalid gate is used."""

    def __init__(self, gate: str) -> None:
        super().__init__(f"Invalid gate: {gate}")


class InvalidParameterError(ValueError):
    """Raised when the number of parameters is invalid."""

    def __init__(self, expected: int, got: int) -> None:
        super().__init__(f"Expected {expected} parameters, got {got}")
