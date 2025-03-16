from dataclasses import dataclass

from paulistringsquantumcircuitsimulations.exceptions import CircuitSystemSizeError


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

    def __init__(self, n: int, instructions: list[Gate] | None = None) -> None:
        """Initialize an empty quantum circuit.

        Args:
            n (int): The number of qubits in the circuit.
            instructions: Optional list of gates to initialize with.

        """
        self.n: int = n
        self.instructions: list[Gate] = instructions if instructions is not None else []

    def append(self, gate: Gate) -> None:
        """Append a gate to the circuit.

        Args:
            gate (Gate): The gate to be appended.

        """
        if max(gate.targets) >= self.n:
            raise CircuitSystemSizeError(max(gate.targets), self.n)

        self.instructions.append(gate)
