from dataclasses import dataclass

import jax.numpy as jnp
from openfermion import get_fermion_operator, jordan_wigner
from openfermion.chem import MolecularData
from openfermionpyscf import run_pyscf

from paulistringsquantumcircuitsimulations.observable import Observable

ComputationalBasisState = str


@dataclass
class HamiltonianInfo:
    n: int
    observables: list[Observable]
    hf_energy: float = 0.0
    hf_state: ComputationalBasisState = ""
    fci_energy: float = 0.0


def get_hamiltonian_info_from_molecule(
    molecule: MolecularData,
    *,
    run_fci: int = 1,
) -> HamiltonianInfo:
    n: int = jnp.int64(molecule.n_orbitals) * 2
    molecule = run_pyscf(molecule, run_scf=1, run_fci=run_fci)

    n_electrons = molecule.n_electrons  # 電子数

    hf_state_list = ["0"] * n
    for i in range(n_electrons):
        hf_state_list[i] = "1"
    hf_state: ComputationalBasisState = "".join(hf_state_list)

    jw_hamiltonian = jordan_wigner(get_fermion_operator(molecule.get_molecular_hamiltonian()))
    observables: list[Observable] = []
    for op, coef in jw_hamiltonian.terms.items():
        pauli_string = ["_"] * n
        for qubit_index, operator in op:
            pauli_string[qubit_index] = operator
        observables.append(Observable(coef, "".join(pauli_string)))
    hf_energy: float = jnp.float64(molecule.hf_energy)
    fci_energy: float = jnp.float64(molecule.fci_energy) if run_fci == 1 else 0.0
    return HamiltonianInfo(n, observables, hf_energy, hf_state, fci_energy)
