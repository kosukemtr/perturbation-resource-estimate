
from distutils.ccompiler import new_compiler
import numpy
import openfermion
from openfermion import MolecularData, InteractionOperator
from pyscf import gto, scf, lo, ao2mo
from openfermionpyscf import run_pyscf
from pyscf.mcscf.PiOS import MakePiOS
from local_hamiltonian import integrals_to_InteractionOp
from openfermion.ops.representations import get_active_space_integrals

def get_PiOS_hamiltonian(mol, *args, **kwargs)-> openfermion.FermionOperator:
    """
    Args:
        mol: molecular geometry pyscf instance
        method:  ['lowdin' | 'meta-lowdin' | 'nao']
    Reference:
        https://sunqm.github.io/pyscf/lo.html#pyscf.lo.orth.orth_ao
    """
    mf = scf.RHF(mol).run()
    nuc_energy = mf.energy_nuc()
    # get index of C's
    c_indices = []
    for i, a in enumerate(mol._atom):
        if a[0] == "C":
            c_indices.append(i+1)
    ncore, nactive, nvirtual, nElec, COrbNew = MakePiOS(mol, mf, c_indices)
    # mat = lo.orth_ao(mf, method, *args, **kwargs) ## this corresponds to mo_coef
    mat = COrbNew

    ## https://github.com/quantumlib/OpenFermion-PySCF/blob/master/openfermionpyscf/_run_pyscf.py  
    one_body_integrals = mat.T.dot( mf.get_hcore() ).dot( mat )

    ## make MO integrals from AO integral. 
    ## https://sunqm.github.io/pyscf/ao2mo.html
    # https://github.com/pyscf/pyscf/blob/master/examples/ao2mo/00-mo_integrals.py
    two_body_integrals = ao2mo.kernel(mol, mat)
    two_body_integrals = ao2mo.restore(1, two_body_integrals, mat.shape[0]) # make 4-leg tensor
    two_body_integrals = numpy.asarray(
        two_body_integrals.transpose(0, 2, 3, 1), order='C')

    whole_hamiltonian = integrals_to_InteractionOp(nuc_energy, one_body_integrals, two_body_integrals)
    
    # compute active space hamiltonian
    e_core, one_body_integrals_new, two_body_integrals_new =\
         get_active_space_integrals(
            one_body_integrals, two_body_integrals,
            occupied_indices=list(range(0, ncore)),
            active_indices=list(range(ncore, ncore+nactive))
        )
    active_space_hamiltonian = integrals_to_InteractionOp(nuc_energy+e_core, one_body_integrals_new, two_body_integrals_new)
    return whole_hamiltonian, active_space_hamiltonian

def get_PiOS_active_hamiltonian(mol, *args, **kwargs)-> openfermion.FermionOperator:
    """
    Args:
        mol: molecular geometry pyscf instance
        method:  ['lowdin' | 'meta-lowdin' | 'nao']
    Reference:
        https://sunqm.github.io/pyscf/lo.html#pyscf.lo.orth.orth_ao
    """
    mf = scf.RHF(mol).run()
    nuc_energy = mf.energy_nuc()
    # get index of C's
    c_indices = []
    for i, a in enumerate(mol._atom):
        if a[0] == "C":
            c_indices.append(i+1)
    ncore, nactive, nvirtual, nElec, COrbNew = MakePiOS(mol, mf, c_indices)
    # mat = lo.orth_ao(mf, method, *args, **kwargs) ## this corresponds to mo_coef
    mat = COrbNew

    ## https://github.com/quantumlib/OpenFermion-PySCF/blob/master/openfermionpyscf/_run_pyscf.py  
    one_body_integrals = mat.T.dot( mf.get_hcore() ).dot( mat )

    ## make MO integrals from AO integral. 
    ## https://sunqm.github.io/pyscf/ao2mo.html
    # https://github.com/pyscf/pyscf/blob/master/examples/ao2mo/00-mo_integrals.py
    two_body_integrals = ao2mo.kernel(mol, mat)
    two_body_integrals = ao2mo.restore(1, two_body_integrals, mat.shape[0]) # make 4-leg tensor
    two_body_integrals = numpy.asarray(
        two_body_integrals.transpose(0, 2, 3, 1), order='C')

    whole_hamiltonian = integrals_to_InteractionOp(nuc_energy, one_body_integrals, two_body_integrals)
    
    # compute active space hamiltonian
    e_core, one_body_integrals_new, two_body_integrals_new =\
         get_active_space_integrals(
            one_body_integrals, two_body_integrals,
            occupied_indices=list(range(0, ncore)),
            active_indices=list(range(ncore, ncore+nactive))
        )
    active_space_hamiltonian = integrals_to_InteractionOp(nuc_energy+e_core, one_body_integrals_new, two_body_integrals_new)
    return whole_hamiltonian, active_space_hamiltonian

def get_PiOS_hamiltonian_integrals(mol, *args, **kwargs)-> openfermion.FermionOperator:
    """
    Args:
        mol: molecular geometry pyscf instance
        method:  ['lowdin' | 'meta-lowdin' | 'nao']
    Reference:
        https://sunqm.github.io/pyscf/lo.html#pyscf.lo.orth.orth_ao
    """
    mf = scf.RHF(mol).run()
    nuc_energy = mf.energy_nuc()
    print("nuc energy", nuc_energy)
    # get index of C's
    c_indices = []
    for i, a in enumerate(mol._atom):
        if a[0] == "C":
            c_indices.append(i+1)
    ncore, nactive, nvirtual, nElec, COrbNew = MakePiOS(mol, mf, c_indices)
    active_start = ncore
    active_end = ncore + nactive
    # mat = lo.orth_ao(mf, method, *args, **kwargs) ## this corresponds to mo_coef
    mat = COrbNew

    ## https://github.com/quantumlib/OpenFermion-PySCF/blob/master/openfermionpyscf/_run_pyscf.py  
    one_body_integrals = mat.T.dot( mf.get_hcore() ).dot( mat )

    ## make MO integrals from AO integral. 
    ## https://sunqm.github.io/pyscf/ao2mo.html
    # https://github.com/pyscf/pyscf/blob/master/examples/ao2mo/00-mo_integrals.py
    two_body_integrals = ao2mo.kernel(mol, mat)
    two_body_integrals = ao2mo.restore(1, two_body_integrals, mat.shape[0]) # make 4-leg tensor
    return active_start, active_end, one_body_integrals, two_body_integrals

