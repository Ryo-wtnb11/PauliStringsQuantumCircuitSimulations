# Pauli Strings Quantum Circuits Simulations

A Python package for simulating quantum circuits based on the Pauli string representation of observables.
It includes the Heisenberg picture simulation of quantum circuits.
It will be extended to the stabilizer formalism simulation (including magic) and PauliString manipulation for physics problems in the future.

## Installation

```bash
pip install paulistringsquantumcircuitsimulations
```

## Usage

```python
from paulistringsquantumcircuitsimulations import Circuit, Gate, Observable, HeisenbergSimulator

# Create a circuit with 2 qubits
circuit = Circuit(n=2)

# Add a Hadamard gate to the circuit
circuit.append(Gate(name="H", targets=[0]))


```
