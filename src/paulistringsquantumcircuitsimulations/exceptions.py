class CircuitSystemSizeError(ValueError):
    """Raised when gate targets exceed system size."""

    def __init__(self, target: int, n: int) -> None:
        super().__init__(f"Qubit index {target} exceeds system size {n}")


class ObservableLengthError(ValueError):
    """Raised when observable length exceeds system size."""

    def __init__(self, length: int, n: int) -> None:
        super().__init__(f"Observable length {length} exceeds system size {n}")
