import random
from dataclasses import dataclass

from paulistringsquantumcircuitsimulations.exceptions import CircuitSystemSizeError


@dataclass
class Gate:
    """Represents a quantum gate operation.

    Args:
        name (str): Name of the gate (e.g. "H", "X", "Y", "Z", "CNOT", "RX", "RY", "RZ", "S")
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
        >>> circuit = Circuit(n=2)
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


def random_single_qubit_clifford_gate(index: int) -> Gate:
    """Generate a random single-qubit Clifford gate.

    Args:
        index (int): The index of the qubit to apply the gate to.

    Returns:
        Gate: A random single-qubit Clifford gate.

    """
    gate_names = [
        "C_NXYZ",
        "C_NZYX",
        "C_XNYZ",
        "C_XYNZ",
        "C_XYZ",
        "C_ZNYX",
        "C_ZYNX",
        "C_ZYX",
        "H",
        "H_NXY",
        "H_NXZ",
        "H_NYZ",
        "H_XY",
        "H_XZ",
        "H_YZ",
        "S",
        "SQRT_X",
        "SQRT_X_DAG",
        "SQRT_Y",
        "SQRT_Y_DAG",
        "SQRT_Z",
        "SQRT_Z_DAG",
        "S_DAG",
    ]
    return Gate(name=random.choice(gate_names), targets=[index])  # noqa: S311
