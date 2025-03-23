import random
from dataclasses import dataclass

import stim

from paulistringsquantumcircuitsimulations.exceptions import CircuitSystemSizeError
from paulistringsquantumcircuitsimulations.utils import PauliString


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
        n_qubits (int):
            The number of qubits in the circuit.
        instructions (list[Gate]):
            List of gates to be executed in the circuit.
            It is optional and be initiallized defaultly to an empty list.

    Examples:
        >>> circuit = Circuit(n_qubits=2)
        >>> circuit.append(Gate(name="H", targets=[0]))
        >>> circuit.append(Gate(name="CNOT", targets=[0, 1]))

    """

    def __init__(self, n_qubits: int, instructions: list[Gate] | None = None) -> None:
        """Initialize a quantum circuit.

        Args:
            n_qubits (int): The number of qubits in the circuit.
            instructions (list[Gate]):
                Optional list of gates to initialize with.
                It is defaultly initialized to an empty list.

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
        r"""Get the sequence of pauli strings and their signs for each rotation (magic) gate in the circuit.

        Let us consider the expectation value of the following circuit:

        $$
            \bra{0} C^{\dagger}_1 U^{\dagger}_1 C^{\dagger}_2 U^{\dagger}_2 \cdots C^{\dagger}_N U^{\dagger}_N
                \mathcal{O} U_N C_N \cdots U_2 C_2 U_1 C_1 \ket{0}
        $$

        where $\mathcal{O}$ is a Pauli operator, and $U_i$ and $C_i$ with $i \in [1, N]$
        are a rotation gate and a Clifford gate, respectively.
        Note that we can rewrite the above circuit as follows:

        $$
        \begin{aligned}
            U_N C_N \cdots U_2 C_2 U_1 C_1 \ket{0} &=
            U_N C_N \cdots U_2 C_2 C_1 (C^{\dagger}_1 U_1 C_1) \ket{0} \\
            &= U_N C_N \cdots C_2 C_1 ( C^{\dagger}_1 C^{\dagger}_2 U_2 C_2 C_1) U'_1 \ket{0} \\
            &= C_N C_{N-1} \cdots C_2 C_1 U'_N \cdots U'_2 U'_1 \ket{0}
        \end{aligned}
        $$

        where $U'_i = \left( \prod_{j=1}^{i} C^{\dagger}_j \right) U_i \left( \prod_{j=i}^{1} C_j \right)$.

        Consequently, the sequence of Pauli strings and signs of $U'_i$ for $i \in [1, N]$ are returned.

        Returns:
            (tuple[list[PauliString], list[complex]]):
                The sequence of Pauli strings and their signs.

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
        r"""Transform Pauli strings after applying all Clifford gates that are applied in the circuit.

        Specifically, it returns

        $$
            C^{\dagger}_1 C^{\dagger}_2 \cdots C^{\dagger}_N \mathcal{O} C_N \cdots C_2 C_1
        $$

        where $\mathcal{O}$ is a Pauli operator that are given as input.
        See the `get_paulistrings` for more details of the transformation.

        Args:
            paulistrings (list[PauliString]): The Pauli strings to be transformed.

        Returns:
            (tuple[list[PauliString], list[complex]]):
                The Pauli strings after applying all Clifford gates.

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


def gate_to_paulistring(n_qubits: int, gate: str, index: list[int]) -> PauliString:
    """Convert a gate to a PauliString.

    Args:
        n_qubits (int): The number of qubits.
        gate (str): The gate to be converted.
        index (int): The index of the qubit to be converted.

    Returns:
        (PauliString): The PauliString representation of the gate.

    """
    gate_symbols = {"Rx": "X", "Ry": "iY", "Rz": "Z"}

    pauli_chars: list[str] = ["_"] * n_qubits
    pauli_chars[index[0]] = gate_symbols.get(gate, "_")
    pauli_str = "".join(pauli_chars)

    return PauliString(pauli_str)


def random_single_qubit_clifford_gate(index: int) -> Gate:
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
