import random
from dataclasses import dataclass

import stim

from paulistringsquantumcircuitsimulations.exceptions import CircuitSystemSizeError
from paulistringsquantumcircuitsimulations.paulioperators import PauliString


@dataclass
class Gate:
    """Represents a gate.

    Args:
        name (str): The name of the gate.
        targets (list[int]): The targets of the gate.

    Examples:
        >>> gate = Gate(name="H", targets=[0])
        >>> gate = Gate(name="CNOT", targets=[0, 1])

    """

    name: str
    targets: list[int]


class Circuit:
    """Represents a quantum circuit.

    Args:
        n (int): The number of qubits in the circuit.
        instructions (list[QuantumGate]): List of gates to be executed

    Examples:
        >>> circuit = Circuit(n=2)
        >>> circuit.append(Gate(name="H", targets=[0]))
        >>> circuit.append(Gate(name="CNOT", targets=[0, 1]))

    """

    def __init__(self, n_qubits: int, instructions: list[Gate] | None = None) -> None:
        """Initialize an empty quantum circuit.

        Args:
            n_qubits (int): The number of qubits in the circuit.
            instructions: Optional list of gates to initialize with.

        """
        self.n_qubits: int = n_qubits
        self.instructions: list[Gate] = instructions if instructions is not None else []
        self.rotation_gate_indices: list[int] = []
        for i, gate in enumerate(self.instructions):
            if gate.name in ["Rx", "Ry", "Rz"]:
                self.rotation_gate_indices.append(i)

    def append(self, gate: Gate) -> None:
        """Append a gate to the circuit.

        Args:
            gate (Gate): The gate to be appended.

        """
        if max(gate.targets) >= self.n_qubits:
            raise CircuitSystemSizeError(max(gate.targets), self.n_qubits)

        self.instructions.append(gate)
        if gate.name in ["Rx", "Ry", "Rz"]:
            self.rotation_gate_indices.append(len(self.instructions) - 1)

    def get_paulistrings(self) -> tuple[list[PauliString], list[complex]]:
        """Get the Pauli strings for each rotation gate in the circuit.

        Returns:
            list[PauliString]: The Pauli strings for each rotation gate in the circuit.

        """
        paulistrings: list[PauliString] = []
        signs: list[complex] = []

        current_cliffords: list[Gate] = []
        last_rot_idx: int = 0

        for rot_idx in self.rotation_gate_indices:
            # Add new Clifford gates since last rotation
            current_cliffords.extend(
                gate
                for i, gate in enumerate(self.instructions[last_rot_idx:rot_idx], start=last_rot_idx)
                if i not in self.rotation_gate_indices
            )

            # Convert rotation to Pauli string and apply Cliffords
            rot_gate = self.instructions[rot_idx]
            pauli = gate_to_paulistring(self.n_qubits, rot_gate.name, rot_gate.targets)

            # Apply accumulated Cliffords
            circuit = stim.Circuit()
            for gate in current_cliffords:
                circuit.append(gate.name, gate.targets)

            pauli_stim = stim.PauliString(pauli).before(circuit)
            paulistrings.append(str(pauli_stim)[-self.n_qubits :])
            signs.append(pauli_stim.sign)
            last_rot_idx = rot_idx + 1

        return paulistrings, signs

    def transform_paulistrings(
        self,
        paulistrings: list[PauliString],
    ) -> tuple[list[PauliString], list[complex]]:
        """Transform Pauli strings to Pauli strings after applying Cliffords that are applied in the circuit.

        Returns:
            list[PauliString]: The Pauli strings after applying Cliffords.

        """
        cliffords: list[Gate] = [
            gate for i, gate in enumerate(self.instructions) if i not in self.rotation_gate_indices
        ]

        circuit = stim.Circuit()
        for gate in cliffords:
            circuit.append(gate.name, gate.targets)

        paulistrings_stim = [stim.PauliString(pauli).before(circuit) for pauli in paulistrings]

        return [str(pauli)[-self.n_qubits :] for pauli in paulistrings_stim], [
            pauli.sign for pauli in paulistrings_stim
        ]


def gate_to_paulistring(n: int, gate: str, index: list[int]) -> PauliString:
    """Convert a gate to a PauliString.

    Args:
        n (int): The number of qubits.
        gate (str): The gate to be converted.
        index (int): The index of the qubit to be converted.

    Returns:
        stim.PauliString: The PauliString representation of the gate.

    """
    gate_symbols = {"Rx": "X", "Ry": "Y", "Rz": "Z"}

    pauli_chars: list[str] = ["_"] * n
    pauli_chars[index[0]] = gate_symbols.get(gate, "_")
    pauli_str = "".join(pauli_chars)

    return PauliString(pauli_str)


def random_single_qubit_clifford_gate(index: int) -> Gate:
    """Generate a random single-qubit Clifford gate.

    Args:
        index (int): The index of the qubit to apply the gate to.

    Returns:
        Gate: A random single-qubit Clifford gate.

    """
    gate_names = [
        "I",
        "X",
        "Y",
        "Z",
        "H",
        "C_XYZ",
        "C_ZYX",
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
