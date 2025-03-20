# PauliStrings Operations for Quantum Circuits Simulations

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

# Create an observable
observable = Observable(coefficient=1.0, paulistring="Z_")

# Create a HeisenbergSimulator
heisenberg_simulator = HeisenbergSimulator(circuit=circuit, observables=[observable])

# Simulate the circuit
observables = heisenberg_simulator.simulate()

# Print the observables
print(observables) # -> [Observable(coefficient=1.0, paulistring=stim.PauliString("+X_"))]
```

## Contact

If you have any questions or feedback, please contact me at [bra](bra).
