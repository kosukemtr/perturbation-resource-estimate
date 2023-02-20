
import numpy as np
import openfermion
import pyscf
from pyscf import gto, scf, lo, ao2mo, fci
from openfermionpyscf import run_pyscf

def local_group_qubit_operator(op, group):
    """
    obtain local group of QubitOperator
    Args:
        op: operator to partition
        group: list of int
    Return:
        QubitOperator containing Pauli Operators acting on the specified group
    """
    ret = openfermion.QubitOperator()
    for term in op.terms:
        in_group = True
        # handle the identity operator
        if len(term) == 0:
            continue
        for pauli in term:
            # pauli is tuple of (index, "P")
            if pauli[0] not in group:
                in_group = False
                break
        if in_group:
            ret += openfermion.QubitOperator(term, op.terms[term])
    return ret

def get_coulomb_like_interaction_group(op):
    """
    get XZZZYIIIZ or ZIIIIZ type interaction operator in op.
    Args:
        op: operator to partition
        group: list of int
    Return:
        QubitOperator containing Pauli Operators in the specific form
    """
    ret = openfermion.QubitOperator()
    for term in op.terms:
        in_group = False
        # handle the identity operator
        if len(term) == 0:
            continue
        XYindex = []
        Zindex = []
        for pauli in term:
            # pauli is tuple of (index, "P")
            if pauli[1] == "X" or pauli[1] == "Y":
                XYindex.append(pauli[0])
            if pauli[1] == "Z":
                Zindex.append(pauli[0])
        # if there is only 2 X/Y
        if len(XYindex) == 2 and len(Zindex) == XYindex[1]-XYindex[0]:
            in_group = True
            # assure Z is in nice position
            # for i in Zindex:
            #     # this condition assures the term is in the form of XZZZZYIIIIZ
            #     # since there is only two X/Y
            #     if i < XYindex[0] or XYindex[1] < i:
            #         assert len(Zindex) == XYindex[1]-XYindex[0]
            #         in_group = True
        if len(XYindex) == 0 and len(Zindex) == 2:
            in_group = True
        if in_group:
            ret += openfermion.QubitOperator(term, op.terms[term])
    return ret

def get_active_space_hamiltonian_with_other_z_terms(op, active_indices):
    """
    obtain active space hamiltonian, but without tracing out the virtual and occupied orbitals
    that is, terms that have X or Y at vir and occ orbitals are removed from op.
    Args:
        op: operator to partition
    Return:
        QubitOperator corresponding to active space hamiltonian
    """
    ret = openfermion.QubitOperator()
    for term in op.terms:
        in_group = True
        # handle the identity operator
        if len(term) == 0:
            ret += op.terms[term]
            continue
        for pauli in term:
            # pauli is tuple of (index, "P")
            if pauli[0] not in active_indices and pauli[1] in ["X", "Y"]:
                in_group = False
                break
        if in_group:
            ret += openfermion.QubitOperator(term, op.terms[term])
    return ret


def _get_hamiltonian_coefs_from_integrals_inner(one_electron_integrals, two_electron_integrals):
    """
    based on simplified version of https://journals.aps.org/prresearch/pdf/10.1103/PhysRevResearch.3.033127 Eq. (C13)
    get numpy array of hamiltonian coefficients
    """
    n_orbs = one_electron_integrals.shape[0]
    identity_coef = 0
    
    identity_coef += np.einsum("pp", one_electron_integrals)
    identity_coef += np.einsum("pprr", two_electron_integrals)/2
    identity_coef += - np.einsum("prrp", two_electron_integrals)/2
    identity_coef += np.einsum("prpr", two_electron_integrals)/4
    
    one_body_majorana_coefs = one_electron_integrals + \
                                np.einsum("pqrr->pq", two_electron_integrals) - \
                                np.einsum("prrq->pq", two_electron_integrals) / 2
    # the below line is required for a correct coef including phases, but for the resource estimation it is not required
    # one_body_majorana_coefs *= 1j
    one_body_majorana_coefs *= 1/2
    
    # for doing index conditioning, prepare ogrid
    p,q,r,s = np.ogrid[:n_orbs,:n_orbs,:n_orbs,:n_orbs]

    # two_body_majorana_coefs_same_spin[p,q,r,s] is the coefficient for majorana in the form of 
    # gamma_p,sigma,0 gamma_r,sigma,0 gamma_q,sigma,1 gamma_s,sigma,0
    # for p>r and s>q
    two_body_majorana_coefs_same_spin = (two_electron_integrals - np.einsum("pqrs->psrq", two_electron_integrals))/4
    two_body_majorana_coefs_same_spin = two_body_majorana_coefs_same_spin[((p > r) & (s > q)).nonzero()]
    # print(two_body_majorana_coefs_same_spin)
    
    # diff spin two body majorana terms
    # take p>r, q<=s terms. This term still have summation of spin configurations. 
    two_body_majorana_coefs_diff_spin_1 = two_electron_integrals[((p>r)&(q<=s)).nonzero()]/4
    # take p>=r, q>s terms. This term still have summation of spin configurations
    two_body_majorana_coefs_diff_spin_2 = two_electron_integrals[((p>=r)&(q>s)).nonzero()]/4
    # take symmetric terms. This term does not need to be summed about spin configurations
    two_body_majorana_coefs_diff_spin_3 = np.einsum("pqpq->pq", two_electron_integrals)/4

    return identity_coef, one_body_majorana_coefs, two_body_majorana_coefs_same_spin, \
        two_body_majorana_coefs_diff_spin_1, \
        two_body_majorana_coefs_diff_spin_2, \
        two_body_majorana_coefs_diff_spin_3

def get_hamiltonian_coefs_from_integrals(one_electron_integrals, two_electron_integrals):
    """
    get coefficients of distinct majorana operators appearing in non-relativistic molecular hamiltonian
    Args:
        one_electron_integrals: 2D numpy array
        two_electron_integrals: 4D numpy array (in chemists notation)
    """
    result = _get_hamiltonian_coefs_from_integrals_inner(one_electron_integrals, two_electron_integrals)
    coef_vec = np.array([])
    for i, coef in enumerate(result):
        if i == 0 or i == len(result)-1:
            coef_vec = np.append(coef_vec, coef)
        else:
            coef_vec = np.append(coef_vec, (coef, coef))
    return np.abs(coef_vec)

def get_active_space_total_hamiltonian_coefs_from_integrals(one_electron_integrals, two_electron_integrals, active_start, active_end):
    """
    obtain active space hamiltonian coefs, but without tracing out the virtual and occupied orbitals
    that is, terms that have X or Y at vir and occ orbitals are removed from op.

    almost copy and paste of _get_hamiltonian_coefs_..._inner, 
    but by defining a conditions we get active space hamiltonian coefs
    """
    n_orbs = one_electron_integrals.shape[0]
    n_active = active_end - active_start
    
    n_orbs = one_electron_integrals.shape[0]
    identity_coef = 0
    identity_coef += np.einsum("pp", one_electron_integrals)
    identity_coef += np.einsum("pprr", two_electron_integrals)/2
    identity_coef += - np.einsum("prrp", two_electron_integrals)/2
    identity_coef += np.einsum("prpr", two_electron_integrals)/4
    
    one_body_coefs = one_electron_integrals + \
                                np.einsum("pqrr->pq", two_electron_integrals) - \
                                np.einsum("prrq->pq", two_electron_integrals) / 2
    # for doing index conditioning, prepare ogrid
    p,q = np.ogrid[:n_orbs,:n_orbs]
    # when p and q are out of active space, take only diagonal terms
    # when p and q are both in active space, take all terms
    condition1 = \
        ((p < active_start)|(active_end <= p)) & (p==q) |\
        (active_start <= p) & (p < active_end) & (active_start <= q) & (q < active_end)
    one_body_coefs = one_body_coefs[condition1.nonzero()]
    one_body_coefs *= 1/2
    
    # for doing index conditioning, prepare ogrid
    p,q,r,s = np.ogrid[:n_orbs,:n_orbs,:n_orbs,:n_orbs]
    p_in_active = (active_start <= p) & (p < active_end)
    q_in_active = (active_start <= q) & (q < active_end)
    r_in_active = (active_start <= r) & (r < active_end)
    s_in_active = (active_start <= s) & (s < active_end)
    p_out_active = (p < active_start) | (active_end <= p)
    q_out_active = (q < active_start) | (active_end <= q)
    r_out_active = (r < active_start) | (active_end <= r)
    s_out_active = (s < active_start) | (active_end <= s)
    
    # two_body_majorana_coefs_same_spin[p,q,r,s] is the coefficient for majorana in the form of 
    # gamma_p,sigma,0 gamma_r,sigma,0 gamma_q,sigma,1 gamma_s,sigma,0
    # for p>r and s>q
    two_body_coefs_same_spin = (two_electron_integrals - np.einsum("pqrs->psrq", two_electron_integrals))/4
    condition2 = \
    ( # take all terms that are in active orbital
        p_in_active & q_in_active & r_in_active & s_in_active
    ) | \
    ( # additionally, if the index is out of active orbitals, take only terms that becomes Z
        p_out_active & r_out_active & (
            # if all pqrs are all out of active
            (p==q) & (r==s) | (p==s) & (r==q) 
        ) |\
        p_out_active & r_in_active & ( (p==q) & s_in_active | (p==s) & q_in_active ) |\
        p_in_active & r_out_active & ( (r==q) & s_in_active | (r==s) & q_in_active) 
    )
    # take the above condition into account, and impose further constraint that (p > r) & (s > q)
    two_body_coefs_same_spin = two_body_coefs_same_spin[((p > r) & (s > q) & condition2).nonzero()]
    
    condition2 = \
    ( # take all terms that are in active orbital
        p_in_active & q_in_active & r_in_active & s_in_active
    ) | \
    ( # additionally, if the index is out of active orbitals, take only terms that becomes Z
        p_out_active & r_out_active & (p == q) & (s == r) |\
        p_out_active & (p == q) & r_in_active & s_in_active |\
        r_out_active & (r == s) & p_in_active & q_in_active
    )
    # diff spin two body majorana terms
    # take p>r, q<=s terms. This term still have summation of spin configurations. 
    two_body_coefs_diff_spin_1 = two_electron_integrals[((p>r)&(q<=s)&condition2).nonzero()]/4
    # print(two_body_coefs_diff_spin_1)
    # take p>=r, q>s terms. This term still have summation of spin configurations
    two_body_coefs_diff_spin_2 = two_electron_integrals[((p>=r)&(q>s)&condition2).nonzero()]/4
    # print(two_body_coefs_diff_spin_2)
    # take symmetric terms. This term does not need to be summed about spin configurations
    # take all diagonal terms and terms that are in active space
    # for doing index conditioning, prepare ogrid
    p,q = np.ogrid[:n_orbs,:n_orbs]
    p_in_active = (active_start <= p) & (p < active_end)
    q_in_active = (active_start <= q) & (q < active_end)
    condition3 = (p==q) | p_in_active & q_in_active
    two_body_coefs_diff_spin_3 = \
        np.einsum("pqpq->pq", two_electron_integrals)[condition3.nonzero()]/4
    
    return identity_coef, one_body_coefs, two_body_coefs_same_spin,\
            two_body_coefs_diff_spin_1, two_body_coefs_diff_spin_2, two_body_coefs_diff_spin_3

def get_inactive_space_total_hamiltonian_coefs_from_integrals(one_electron_integrals, two_electron_integrals, active_start, active_end):
    """
    obtain active space hamiltonian coefs, but without tracing out the virtual and occupied orbitals
    that is, terms that have X or Y at vir and occ orbitals are removed from op.

    almost copy and paste of _get_hamiltonian_coefs_..._inner, 
    but by defining a conditions we get active space hamiltonian coefs
    """
    n_orbs = one_electron_integrals.shape[0]
    n_active = active_end - active_start
    
    n_orbs = one_electron_integrals.shape[0]
    identity_coef = 0
    identity_coef += np.einsum("pp", one_electron_integrals)
    identity_coef += np.einsum("pprr", two_electron_integrals)/2
    identity_coef += - np.einsum("prrp", two_electron_integrals)/2
    identity_coef += np.einsum("prpr", two_electron_integrals)/4
    
    one_body_coefs = one_electron_integrals + \
                                np.einsum("pqrr->pq", two_electron_integrals) - \
                                np.einsum("prrq->pq", two_electron_integrals) / 2
    # for doing index conditioning, prepare ogrid
    p,q = np.ogrid[:n_orbs,:n_orbs]
    # when p and q are out of active space, take only diagonal terms
    # when p and q are both in active space, take all terms
    condition1 = \
        ((p < active_start)|(active_end <= p)) & (p==q) |\
        (active_start <= p) & (p < active_end) & (active_start <= q) & (q < active_end)
    # take not of condition 1
    condition1 = condition1 ^ ((p==p)|(q==q))
    one_body_coefs = one_body_coefs[condition1.nonzero()]
    one_body_coefs *= 1/2
    
    # for doing index conditioning, prepare ogrid
    p,q,r,s = np.ogrid[:n_orbs,:n_orbs,:n_orbs,:n_orbs]
    p_in_active = (active_start <= p) & (p < active_end)
    q_in_active = (active_start <= q) & (q < active_end)
    r_in_active = (active_start <= r) & (r < active_end)
    s_in_active = (active_start <= s) & (s < active_end)
    p_out_active = (p < active_start) | (active_end <= p)
    q_out_active = (q < active_start) | (active_end <= q)
    r_out_active = (r < active_start) | (active_end <= r)
    s_out_active = (s < active_start) | (active_end <= s)
    
    # two_body_majorana_coefs_same_spin[p,q,r,s] is the coefficient for majorana in the form of 
    # gamma_p,sigma,0 gamma_r,sigma,0 gamma_q,sigma,1 gamma_s,sigma,0
    # for p>r and s>q
    two_body_coefs_same_spin = (two_electron_integrals - np.einsum("pqrs->psrq", two_electron_integrals))/4
    condition2 = \
    ( # take all terms that are in active orbital
        p_in_active & q_in_active & r_in_active & s_in_active
    ) | \
    ( # additionally, if the index is out of active orbitals, take only terms that becomes Z
        p_out_active & r_out_active & (
            # if all pqrs are all out of active
            (p==q) & (r==s) | (p==s) & (r==q) 
        ) |\
        p_out_active & r_in_active & ( (p==q) & s_in_active | (p==s) & q_in_active ) |\
        p_in_active & r_out_active & ( (r==q) & s_in_active | (r==s) & q_in_active) 
    )
    condition2 = condition2 ^ ((p==p)|(q==q)|(r==r)|(s==s))
    # take the above condition into account, and impose further constraint that (p > r) & (s > q)
    two_body_coefs_same_spin = two_body_coefs_same_spin[((p > r) & (s > q) & condition2).nonzero()]
    
    condition2 = \
    ( # take all terms that are in active orbital
        p_in_active & q_in_active & r_in_active & s_in_active
    ) | \
    ( # additionally, if the index is out of active orbitals, take only terms that becomes Z
        p_out_active & r_out_active & (p == q) & (s == r) |\
        p_out_active & (p == q) & r_in_active & s_in_active |\
        r_out_active & (r == s) & p_in_active & q_in_active
    )
    condition2 = condition2 ^ ((p==p)|(q==q)|(r==r)|(s==s))
    # diff spin two body majorana terms
    # take p>r, q<=s terms. This term still have summation of spin configurations. 
    two_body_coefs_diff_spin_1 = two_electron_integrals[((p>r)&(q<=s)&condition2).nonzero()]/4
    # print(two_body_coefs_diff_spin_1)
    # take p>=r, q>s terms. This term still have summation of spin configurations
    two_body_coefs_diff_spin_2 = two_electron_integrals[((p>=r)&(q>s)&condition2).nonzero()]/4
    # print(two_body_coefs_diff_spin_2)
    # take symmetric terms. This term does not need to be summed about spin configurations
    # take all diagonal terms and terms that are in active space
    # for doing index conditioning, prepare ogrid
    p,q = np.ogrid[:n_orbs,:n_orbs]
    p_in_active = (active_start <= p) & (p < active_end)
    q_in_active = (active_start <= q) & (q < active_end)
    condition3 = (p==q) | p_in_active & q_in_active
    condition3 = condition3 ^ ((p==p)|(q==q))
    two_body_coefs_diff_spin_3 = \
        np.einsum("pqpq->pq", two_electron_integrals)[condition3.nonzero()]/4
    
    return identity_coef, one_body_coefs, two_body_coefs_same_spin,\
            two_body_coefs_diff_spin_1, two_body_coefs_diff_spin_2, two_body_coefs_diff_spin_3
