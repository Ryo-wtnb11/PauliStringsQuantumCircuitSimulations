from dataclasses import dataclass


@dataclass
class Gate:
    """Represents a quantum gate operation.

    Args:
        name (str): Name of the gate (e.g. "H", "X", "Y", "Z", "CNOT", "RX", "RY", "RZ", "S", "T")
        targets (list[int]): List of target qubit indices the gate operates on

    Note:
        For CNOT gates, control and target qubits are indexed as 0 and 1 respectively,
        following the convention of the stim library.

    Examples:
        >>> gate = Gate(name="H", targets=[0])  # Hadamard gate
        >>> cnot = Gate(name="CNOT", targets=[0, 1])  # CNOT gate

    """

    name: str
    targets: list[int]
    parameter: float = 0.0


class Circuit:
    """Represents a quantum circuit.

    Args:
        instructions (list[Gate]): List of gates to be executed

    Examples:
        >>> circuit = Circuit()
        >>> circuit.append(Gate(name="H", targets=[0]))
        >>> circuit.append(Gate(name="CNOT", targets=[0, 1]))

    """

    def __init__(self) -> None:
        """Initialize an empty quantum circuit.

        The circuit starts with no instructions. Gates can be added using the append method.
        """
        self.instructions: list[Gate] = []

    def append(self, gate: Gate) -> None:
        self.instructions.append(gate)
