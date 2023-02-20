"""
Male second-quantized Hamiltonian by using localized molecular orbitals
lo documentation: https://sunqm.github.io/pyscf/lo.html
"""

import numpy
import openfermion
from openfermion import MolecularData, InteractionOperator
from pyscf import gto, scf, lo, ao2mo
from openfermionpyscf import run_pyscf

EQ_TOLERANCE=1e-11

def get_localized_hamiltonian(mol, method, *args, **kwargs)-> openfermion.FermionOperator:
    """
    Args:
        mol: molecular geometry pyscf instance
        method:  ['lowdin' | 'meta-lowdin' | 'nao']
    Reference:
        https://sunqm.github.io/pyscf/lo.html#pyscf.lo.orth.orth_ao
    """
    mf = scf.RHF(mol).run()
    nuc_energy = mf.energy_nuc()
    mat = lo.orth_ao(mf, method, *args, **kwargs) ## this corresponds to mo_coef

    ## https://github.com/quantumlib/OpenFermion-PySCF/blob/master/openfermionpyscf/_run_pyscf.py  
    one_body_integrals = mat.T.dot( mf.get_hcore() ).dot( mat )

    ## make MO integrals from AO integral. 
    ## https://sunqm.github.io/pyscf/ao2mo.html
    # https://github.com/pyscf/pyscf/blob/master/examples/ao2mo/00-mo_integrals.py
    two_body_integrals = ao2mo.kernel(mol, mat)
    two_body_integrals = ao2mo.restore(1, two_body_integrals, mat.shape[0]) # make 4-leg tensor
    two_body_integrals = numpy.asarray(
        two_body_integrals.transpose(0, 2, 3, 1), order='C')

    op = integrals_to_InteractionOp(nuc_energy, one_body_integrals, two_body_integrals)
    return op

def get_localized_integrals(mol, method, *args, **kwargs)-> openfermion.FermionOperator:
    """
    Args:
        mol: molecular geometry pyscf instance
        method:  ['lowdin' | 'meta-lowdin' | 'nao']
    Reference:
        https://sunqm.github.io/pyscf/lo.html#pyscf.lo.orth.orth_ao
    """
    mf = scf.RHF(mol).run()
    nuc_energy = mf.energy_nuc()
    mat = lo.orth_ao(mf, method, *args, **kwargs) ## this corresponds to mo_coef

    ## https://github.com/quantumlib/OpenFermion-PySCF/blob/master/openfermionpyscf/_run_pyscf.py  
    one_body_integrals = mat.T.dot( mf.get_hcore() ).dot( mat )

    ## make MO integrals from AO integral. 
    ## https://sunqm.github.io/pyscf/ao2mo.html
    # https://github.com/pyscf/pyscf/blob/master/examples/ao2mo/00-mo_integrals.py
    two_body_integrals = ao2mo.kernel(mol, mat)
    two_body_integrals = ao2mo.restore(1, two_body_integrals, mat.shape[0]) # make 4-leg tensor
    two_body_integrals = numpy.asarray(
        two_body_integrals.transpose(0, 2, 3, 1), order='C')

    return nuc_energy, one_body_integrals, two_body_integrals

def integrals_to_InteractionOp(nuc_energy, one_body_integrals, two_body_integrals)-> openfermion.InteractionOperator:
    """ make molecular Hamiltonian from MO integrals.
    Args:
        one_body_integrals: numpy.ndarray((n,n)) coefficients of one body interactions 
        two_body_integrals: numpy.ndarray((n,n,n,n)) coefficients of two body interactions
    Returns:
        molecular_hamiltonian (openfermion.InteractionOperator)
        
    copy-and-pasted from https://github.com/quantumlib/OpenFermion/blob/master/src/openfermion/hamiltonians/_molecular_data.py#L823
    ***above link is outdated; updated link https://github.com/quantumlib/OpenFermion/blob/ac7501b61f42c627b5293abd2e72a91ba527ca5e/src/openfermion/chem/molecular_data.py#L222
    """
    # Initialize Hamiltonian coefficients.
    constant = nuc_energy 
    n_qubits = 2 * one_body_integrals.shape[0]
    one_body_coefficients = numpy.zeros((n_qubits, n_qubits))
    two_body_coefficients = numpy.zeros((n_qubits, n_qubits,
                                            n_qubits, n_qubits))
    # Loop through integrals.
    for p in range(n_qubits // 2):
        for q in range(n_qubits // 2):

            # Populate 1-body coefficients. Require p and q have same spin.
            one_body_coefficients[2 * p, 2 * q] = one_body_integrals[
                p, q]
            one_body_coefficients[2 * p + 1, 2 *
                                    q + 1] = one_body_integrals[p, q]
            # Continue looping to prepare 2-body coefficients.
            for r in range(n_qubits // 2):
                for s in range(n_qubits // 2):

                    # Mixed spin
                    two_body_coefficients[2 * p, 2 * q + 1,
                                            2 * r + 1, 2 * s] = (
                        two_body_integrals[p, q, r, s] / 2.)
                    two_body_coefficients[2 * p + 1, 2 * q,
                                            2 * r, 2 * s + 1] = (
                        two_body_integrals[p, q, r, s] / 2.)

                    # Same spin
                    two_body_coefficients[2 * p, 2 * q,
                                            2 * r, 2 * s] = (
                        two_body_integrals[p, q, r, s] / 2.)
                    two_body_coefficients[2 * p + 1, 2 * q + 1,
                                            2 * r + 1, 2 * s + 1] = (
                        two_body_integrals[p, q, r, s] / 2.)

    # Truncate.
    one_body_coefficients[
        numpy.absolute(one_body_coefficients) < EQ_TOLERANCE] = 0.
    two_body_coefficients[
        numpy.absolute(two_body_coefficients) < EQ_TOLERANCE] = 0.

    # Cast to InteractionOperator class and return.
    molecular_hamiltonian = InteractionOperator(
        constant, one_body_coefficients, two_body_coefficients)

    return molecular_hamiltonian
