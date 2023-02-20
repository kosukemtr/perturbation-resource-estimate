
from cmath import exp
import openfermion
import scipy as sp
from scipy.special import lambertw
import numpy as np

def get_cost_for_first_order_pert(h1norm, v23norm, v1norm, delta0, delta1, Delta, p, r_scale=1.0, epsilon_scale=1.0, n_subsystems=1):
    """
    v23norm = (\sum v^{2/3})^{3/2}
    """
    epsilon = delta1/(20*v1norm)*np.sqrt(p/(1-p))*epsilon_scale
    r = delta1/(20*v1norm)*r_scale
    M1 = 5*np.sqrt(2)/2 * \
            np.e**2/(np.e-1) * \
                v23norm
    kappa = get_kappa(Delta, delta0, h1norm)
    x_th = get_x_th(delta0, h1norm)
    n_filter = get_n_filter(epsilon, kappa, x_th)
    print("epsilon: ", epsilon)
    print("n_filter: ", n_filter)
    print("kappa: ", kappa)
    print("M1: ", M1)
    print("r: ", r)
    print("x_th:", x_th)
    return 2 * M1 * (np.log(2/r)/np.sqrt(p)*n_subsystems) * n_filter

def get_cost_for_second_order_pert(h1norm, v23norm, v1norm, delta0, delta2, Delta, p, r_scale=1.0, epsilon_scale=1.0, n_subsystems=1):
    """
    v23norm = (\sum v^{2/3})^{3/2}
    """
    epsilon_filter = delta2*Delta/(20*v1norm**2)*np.sqrt(p/(1-p))*epsilon_scale
    r = delta2*Delta/(20*v1norm**2)*r_scale
    epsilon_ptb = h1norm/v1norm**2 * delta2 / 10
    w = (Delta-delta0)/h1norm
    w0 = delta0/h1norm
    M2 = 5*np.sqrt(2) * \
            np.e**2/(np.e-1) * \
                1/(w*h1norm) / delta2 * \
                    v23norm**2
    n_filter = get_n_filter(epsilon_filter, get_kappa(Delta, delta0, h1norm),  get_x_th(delta0, h1norm))
    n_ptb = get_n_ptb(epsilon_ptb, w, w0)
    print("r: ", r)
    print("epsilon_filter: ", epsilon_filter)
    print("epsilon_ptb: ", epsilon_ptb)
    print("kappa: ", get_kappa(Delta, delta0, h1norm))
    print("w: ", w)
    print("w0: ", w0)
    print("M2: ", M2)
    print("n_filter: ", n_filter)
    print("n_ptb: ", n_ptb)
    return M2 * (2 * (np.log(2/r)/np.sqrt(p)*n_subsystems) * n_filter + n_ptb)

def get_n_ptb(epsilon, w, w0):
    def b(_epsilon, _w):
        return np.ceil(np.log(1/_w/_epsilon)/_w**2)
    def D(_epsilon, _w):
        return np.ceil(
            np.sqrt(
                b(_epsilon, _w) * \
                np.log(4*b(_epsilon,_w)/_epsilon)
            )
        )
    def n_sign(_epsilon, kappa, c):
        return \
            64 * (1+np.abs(c)) / (np.sqrt(np.pi)*_epsilon) * \
            1/kappa * \
            np.sqrt(2*np.log(8/(np.pi*_epsilon**2))) * \
            np.exp(- 1 / 2 * sp.special.lambertw(2048/(np.pi*_epsilon**2*np.e**2)))
        
    epsilon_primeprime = np.min([
        2*epsilon*w/5,
        1/(4*w*D(epsilon/4, w/2)),
        epsilon/(
            2*w0*(
                D(epsilon/4, w/2) + 1
            )**2
        )
    ])
    return 2 * D(epsilon/4, w/2) + n_sign(epsilon_primeprime, w/4, 3*w/4)

def get_n_filter(epsilon, kappa, x_th):
    return \
        64 * (1+x_th+kappa/2) / (np.sqrt(np.pi)*epsilon) * \
        1/kappa * \
        np.sqrt(2*np.log(8/(np.pi*epsilon**2))) * \
        np.exp(- 1 / 2 * sp.special.lambertw(2048/(np.pi*epsilon**2*np.e**2)))

def get_x_th(delta0, h1norm):
    return delta0/h1norm

def get_kappa(Delta, delta0, h1norm):
    return (Delta-delta0)/h1norm
