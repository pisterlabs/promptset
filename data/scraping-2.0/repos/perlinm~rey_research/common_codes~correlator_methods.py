#!/usr/bin/env python3

# FILE CONTENTS: methods for computing collective spin correlators

import itertools, scipy
import numpy as np

from scipy.integrate import solve_ivp

from special_functions import *


##########################################################################################
# exact OAT results
##########################################################################################

# exact correlators for OAT with decoherence
# adapted from results in foss-feig2013nonequilibrium
def correlators_OAT(spin_num, times, dec_rates = (0,0,0)):
    N = spin_num
    S = N/2
    t = times
    D_p, D_z, D_m = dec_rates

    gamma = ( D_p + D_m ) / 2
    delta = D_p - D_m
    kappa = sum(dec_rates)/2

    if D_m != 0 or D_p != 0:
        Sz_unit = (D_p-D_m)/(D_p+D_m) * (1-np.exp(-(D_p+D_m)*t))
    else:
        Sz_unit = np.zeros(len(times))
    Sz = S * Sz_unit
    Sz_Sz = S/2 + S*(S-1/2) * Sz_unit**2

    gamma_t = gamma*t
    def w(x): return np.sqrt(x**2 - gamma**2 - 1j*x*delta)
    def Phi(x):
        return np.exp(-gamma_t) * ( np.cos(w(x)*t) + gamma_t * np.sinc(w(x)*t/np.pi) )
    def Psi(x):
        return np.exp(-gamma_t) * (delta+1j*x)*t * np.sinc(w(x)*t/np.pi)

    Sp = S * np.exp(-kappa*t) * Phi(1)**(N-1)
    Sp_Sz = -1/2 * Sp + S * (S-1/2) * np.exp(-kappa*t) * Psi(1) * Phi(1)**(N-2)
    Sp_Sp = S * (S-1/2) * np.exp(-2*kappa*t) * Phi(2)**(N-2)
    Sp_Sm = S + Sz + S * (S-1/2) * np.exp(-2*kappa*t)

    return { (0,1,0) : Sz,
             (0,2,0) : Sz_Sz,
             (1,0,0) : Sp,
             (2,0,0) : Sp_Sp,
             (1,1,0) : Sp_Sz,
             (1,0,1) : Sp_Sm }


##########################################################################################
# expectation values
##########################################################################################

# logarithm of the magnitude of < Z | S_-^ll S_\z^mm S_+^nn | Z >
def op_ln_val_Z_m(op, NN, vals = {}):
    ll, mm, nn = op
    if ll != 0 or nn != 0: return None
    val = mm*np.log(NN/2)
    sign = 1 if ( mm % 2 == 0 ) else -1
    vals[mm,NN] = val, sign
    return val, sign

# logarithm of the magnitude of < Z | S_+^ll S_\z^mm S_-^nn | Z >
def op_ln_val_Z_p(op, NN, vals = {}):
    try: return vals[ll,mm,NN]
    except: None

    ll, mm, nn = op
    if ll != nn: return None
    if ll > NN: return None

    ln_factorials_num = ln_factorial(NN) + ln_factorial(ll)
    ln_factorials_den = ln_factorial(NN-ll)
    val = mm*np.log(abs(NN/2-ll)) + ln_factorials_num - ln_factorials_den
    sign = 1 if ( NN/2 > ll or mm % 2 == 0 ) else -1

    vals[ll,mm,NN] = val, sign
    return val, sign

# natural logarithm of factors which appear in the general expectation value
def ln_factorial_factors(NN, kk, ll, nn):
    return ( ln_factorial(NN - kk)
             - ln_factorial(kk)
             - ln_factorial(NN - ll - kk)
             - ln_factorial(NN - nn - kk) )
def ln_polar_factors(NN, mu, ll, nn, kk, theta):
    delta = lambda a, b : 1 if a == b else 0
    cos_term = (2*NN*delta(mu,-1) + mu*(2*kk+ll+nn)) * np.log(np.cos(theta/2))
    sin_term = (2*NN*delta(mu,+1) - mu*(2*kk+ll+nn)) * np.log(np.sin(theta/2))
    return cos_term + sin_term
ln_factorial_factors = np.vectorize(ln_factorial_factors)
ln_polar_factors = np.vectorize(ln_polar_factors)

# logarithm of the magnitude of < \theta,\phi | S_+^ll S_\z^mm S_-^nn | \theta,\phi >
def op_ln_val_ZXY(op, NN, theta = np.pi/2, phi = 0, mu = 1, vals = {}):
    assert(theta >= 0 and theta <= np.pi)

    ll, mm, nn = op
    try:
        val, sign = vals[ll,mm,nn,NN,theta,mu]
        return val, sign * my_expi(phi)**(mu*(ll-nn))
    except: None
    try:
        val, sign = vals[nn,mm,ll,NN,theta,mu]
        return val, sign * my_expi(phi)**(mu*(ll-nn))
    except: None

    if (ll, mm % 2, nn) == (0,1,0) and theta == np.pi/2: return None
    if max(ll,nn) > NN: return None

    if theta == np.pi/2:
        ln_prefactor = ln_factorial(NN) - NN*np.log(2)
        def ln_factors(kk):
            return ln_factorial_factors(NN,kk,ll,nn)
    else:
        ln_prefactor = ln_factorial(NN)
        def ln_factors(kk):
            return ( ln_factorial_factors(NN,kk,ll,nn) +
                     ln_polar_factors(NN,mu,ll,nn,kk,theta) )

    kk_vals = np.arange(NN-max(ll,nn)+0.5, dtype = int)

    # remove kk == NN/2 term if necessary
    if mm > 0 and NN % 2 == 0 and kk_vals[-1] >= NN/2:
        kk_vals = np.delete(kk_vals, NN//2)

    # compute the logarithm of the magnitude of each term
    ln_terms = ln_factors(kk_vals) + ln_prefactor
    offset_kk_vals = kk_vals-NN/2
    if mm != 0: ln_terms += mm * np.log(abs(offset_kk_vals))

    # compute the absolute value of terms divided by the largest term
    ln_term_max = ln_terms.max()
    terms = np.exp(ln_terms-ln_term_max)

    # compute the logarithm of the sum of the terms
    if mm % 2 == 1:
        term_sum = np.sum(np.sign(offset_kk_vals)*terms)
        val = ln_term_max + np.log(abs(term_sum))
        sign = np.sign(term_sum)
    else:
        val = ln_term_max + np.log(np.sum(terms))
        sign = 1

    vals[ll,mm,nn,NN,theta,mu] = val, sign
    return val, sign * my_expi(phi)**(mu*(ll-nn))

# return functions to compute initial values
def init_ln_val_function(spin_num, init_state, mu = 1):
    if type(init_state) is str: init_state = axis_str(init_state)
    z, x, y = 0, 1, 2

    if init_state[x] == 0 and init_state[y] == 0:
        if mu == np.sign(init_state[z]):
            return lambda op : op_ln_val_Z_p(op, spin_num)
        else:
            return lambda op : op_ln_val_Z_m(op, spin_num)

    else:
        angle_zx = np.arccos(init_state[z]/np.linalg.norm(init_state))
        angle_xy = np.arctan2(init_state[y],init_state[x])
        return lambda op : op_ln_val_ZXY(op, spin_num, angle_zx, angle_xy, mu)


##########################################################################################
# machinery for manipulating operator vectors
##########################################################################################

# clean up a dictionary vector
def clean(vec, spin_num = None):
    null_ops = [ op for op, val in vec.items() if abs(val) == 0 ]
    for op in null_ops: del vec[op]
    if spin_num is not None:
        overflow_ops = [ op for op in vec.keys()
                         if op[0] > spin_num or op[2] > spin_num ]
        for op in overflow_ops: del vec[op]
    return vec

# take hermitian conjugate of a dictionary taking operator --> value,
#   i.e. return a dictionary taking operator* --> value*
def conj_vec(vec):
    return { op[::-1] : np.conj(val) for op, val in vec.items() }

# add the right vector to the left vector
def add_left(vec_left, vec_right, scalar = 1):
    for op, val in vec_right.items():
        try:
            vec_left[op] += scalar * val
        except:
            vec_left[op] = scalar * val

# return sum of all input vectors
def sum_vecs(vecs, factors = None):
    if factors is None:
        factors = [1] * len(vecs)
    vec_sum = {}
    for vec, factor in zip(vecs, factors):
        add_left(vec_sum, vec, factor)
    return vec_sum

# return vector S_\mu^ll (x + \mu S_\z)^mm * S_\nu^nn
def binom_op(ll, mm, nn, x, prefactor = 1):
    return { (ll,kk,nn) : prefactor * x**(mm-kk) * binom(mm,kk) for kk in range(mm+1) }

# takes S_\mu^ll (\mu S_\z)^mm S_\nu^nn
#   --> S_\mu^ll [ \sum_jj x_jj (\mu S_\z)^jj ] (\mu S_\z)^mm + S_\nu^nn
def insert_z_poly(vec, coefficients, prefactor = 1):
    output = { op : prefactor * coefficients[0] * val for op, val in vec.items() }
    for jj in range(1,len(coefficients)):
        for op, val in vec.items():
            ll, mm, nn = op
            try:
                output[ll,mm+jj,nn] += prefactor * coefficients[jj] * val
            except:
                output[ll,mm+jj,nn] = prefactor * coefficients[jj] * val
    return output

# shorthand for operator term: "extended binomial operator"
def ext_binom_op(ll, mm, nn, terms, x, prefactor = 1):
    return insert_z_poly(binom_op(ll,mm,nn,x), terms, prefactor)


##########################################################################################
# general commutator between ordered products of collective spin operators
##########################################################################################

# simplify product of two operators
def multiply_terms(op_left, op_right):
    pp, qq, rr = op_left
    ll, mm, nn = op_right
    vec = {}
    binom_qq = [ binom(qq,bb) for bb in range(qq+1) ]
    binom_mm = [ binom(mm,cc) for cc in range(mm+1) ]
    for kk in range(min(rr,ll)+1):
        kk_fac = factorial(kk) * binom(rr,kk) * binom(ll,kk)
        for aa, bb, cc in itertools.product(range(kk+1),range(qq+1),range(mm+1)):
            bb_fac = (ll-kk)**(qq-bb) * binom_qq[bb]
            cc_fac = (rr-kk)**(mm-cc) * binom_mm[cc]
            kabc_fac = kk_fac * zeta(rr,ll,kk,aa) * bb_fac * cc_fac
            op_in = (pp+ll-kk, aa+bb+cc, rr+nn-kk)
            try:
                vec[op_in] += kabc_fac
            except:
                vec[op_in] = kabc_fac
    return clean(vec)

# simplify product of two vectors
def multiply_vecs(vec_lft, vec_rht, prefactor = 1, dag_lft = False, dag_rht = False):
    if dag_lft: vec_lft = conj_vec(vec_lft)
    if dag_rht: vec_rht = conj_vec(vec_rht)
    vec = {}
    for term_lft, val_lft in vec_lft.items():
        for term_rht, val_rht in vec_rht.items():
            fac = val_lft * val_rht * prefactor
            add_left(vec, multiply_terms(term_lft, term_rht), fac)
    return clean(vec)


##########################################################################################
# miscellaneous methods for changing frames and operator vectors
##########################################################################################

# decoherence transformation matrix from a periodic drive; here A = J_0(\beta), where:
#   J_0 is the zero-order bessel function of the first kind
#   \beta is the modulation index
def dec_mat_drive(A, mu = 1): # in (mu,z,bmu) format
    const = np.array([[ 1, 0, 1 ],
                      [ 0, 0, 0 ],
                      [ 1, 0, 1 ]]) * 1/2
    var = np.array([[  mu, 0, -mu ],
                    [   0, 2,   0 ],
                    [ -mu, 0,  mu ]]) * 1/2
    return const + A * var

# convert 3D vectors between (z,x,y) and (mu,z,bmu) formats
def pzm_to_zxy_mat(mu = 1):
    return np.array([ [      0, 1,      0 ],
                      [      1, 0,      1 ],
                      [ +mu*1j, 0, -mu*1j ] ])

def mat_zxy_to_pzm(mat_zxy, mu = 1):
    pzm_to_zxy = pzm_to_zxy_mat(mu)
    zxy_to_pzm = np.linalg.inv(pzm_to_zxy)
    return zxy_to_pzm @ mat_zxy @ pzm_to_zxy

def mat_pzm_to_zxy(mat_pzm, mu = 1):
    pzm_to_zxy = pzm_to_zxy_mat(mu)
    zxy_to_pzm = np.linalg.inv(pzm_to_zxy)
    return pzm_to_zxy @ mat_pzm @ zxy_to_pzm

# convert operator vectors between (z,x,y) and (mu,z,bmu) formats
def vec_zxy_to_pzm(vec_zxy, negative_z_direction = None, mu = 1):
    Sz = { (0,1,0) : 1 }
    Sx = { (1,0,0) : 1/2,
           (0,0,1) : 1/2 }
    Sy = { (1,0,0) : -mu*1j/2,
           (0,0,1) : +mu*1j/2 }
    if negative_z_direction is not None:
        if type(negative_z_direction) is str:
            negative_z_direction = axis_str(negative_z_direction)
        theta, phi = vec_theta_phi(-np.array(negative_z_direction))
        Sz_new = sum_vecs([ Sz, Sx, Sy ], [ my_cos(theta),
                                            my_sin(theta) * my_cos(phi),
                                            my_sin(theta) * my_sin(phi) ])
        Sx_new = sum_vecs([ Sz, Sx, Sy ], [ -my_sin(theta),
                                            my_cos(theta) * my_cos(phi),
                                            my_cos(theta) * my_sin(phi) ])
        Sy_new = sum_vecs([ Sx, Sy ], [ -my_sin(phi), my_cos(phi) ])
        Sz, Sx, Sy = Sz_new, Sx_new, Sy_new

    vec_pzm = {} # operator vector in (mu,z,bmu) format
    for op_zxy, val_zxy in vec_zxy.items():
        ll, mm, nn = op_zxy
        # starting from the left, successively multiply all factors on the right
        lmn_vec = { (0,0,0) : 1 }
        for jj in range(ll): lmn_vec = multiply_vecs(lmn_vec, Sz)
        for jj in range(mm): lmn_vec = multiply_vecs(lmn_vec, Sx)
        for jj in range(nn): lmn_vec = multiply_vecs(lmn_vec, Sy)
        add_left(vec_pzm, lmn_vec, val_zxy * mu**ll)
    if np.array([ np.imag(val) == 0 for val in vec_pzm.values() ]).all():
        vec_pzm = { op : np.real(val) for op, val in vec_pzm.items() }
    return clean(vec_pzm)

# given correlators of the form < S_\mu^ll (\mu S_z)^mm S_\bmu^nn >,
#   convert them into correlators of the form < S_\bmu^ll (\bmu S_z)^mm S_\mu^nn >
def invert_vals(vals):
    inverted_vals = {}
    target_ops = list(vals.keys())
    for ll, mm, nn in target_ops:
        if vals.get((nn,mm,ll)) is None:
            vals[nn,mm,ll] = np.conj(vals[ll,mm,nn])
    for ll, mm, nn in target_ops:
        coeffs_mm_nn = multiply_terms((0,mm,0),(nn,0,0))
        coeffs_ll_mm_nn = multiply_vecs({(0,0,ll):1}, coeffs_mm_nn, (-1)**mm)
        inverted_vals[ll,mm,nn] \
            = sum([ coeff * vals[op] for op, coeff in coeffs_ll_mm_nn.items()
                    if vals.get(op) is not None ])
    return inverted_vals


##########################################################################################
# single-spin decoherence
##########################################################################################

# diagonal terms of single-spin decoherence
def op_image_decoherence_diag_individual(op, SS, dec_vec, mu):
    ll, mm, nn = op
    D_p, D_z, D_m = abs(np.array(dec_vec))**2
    if mu == 1:
        D_mu, D_nu = D_p, D_m
    else:
        D_mu, D_nu = D_m, D_p

    image_mu = {}
    if D_mu != 0:
        image_mu = ext_binom_op(*op, [ SS-ll-nn, -1 ], 1, D_mu)
        add_left(image_mu, insert_z_poly({op:1}, [ SS-(ll+nn)/2, -1 ]), -D_mu)
        if ll >= 1 and nn >= 1:
            image_mu[ll-1, mm, nn-1] = ll*nn * (2*SS-ll-nn+2) * D_mu
        if ll >= 2 and nn >= 2:
            op_2 = (ll-2, mm, nn-2)
            factor = ll*nn*(ll-1)*(nn-1)
            image_mu.update(ext_binom_op(*op_2, [ SS, 1 ], -1, factor * D_mu))

    image_nu = {}
    if D_nu != 0:
        image_nu = ext_binom_op(*op, [ SS, 1 ], -1, D_nu)
        add_left(image_nu, insert_z_poly({op:1}, [ SS+(ll+nn)/2, 1 ]), -D_nu)

    image_z = {}
    if D_z != 0 and ll + nn != 0:
        image_z = { (ll,mm,nn) : -1/2*(ll+nn) * D_z }
        if ll >= 1 and nn >= 1:
            image_z.update(ext_binom_op(ll-1, mm, nn-1, [ SS, 1 ], -1, ll*nn * D_z))

    return sum_vecs([ image_mu, image_nu, image_z ])

# single-spin decoherence "Q" cross term
def op_image_decoherence_Q_individual(op, SS, dec_vec, mu):
    ll, mm, nn = op
    g_p, g_z, g_m = dec_vec
    if mu == 1:
        g_mu, g_nu = g_p, g_m
    else:
        g_mu, g_nu = g_m, g_p

    gg_mp = np.conj(g_nu) * g_mu
    gg_zp = np.conj(g_z) * g_mu
    gg_mz = np.conj(g_nu) * g_z

    image_P = {}
    if gg_mp != 0 and nn != 0:
        if nn >= 2:
            image_P = ext_binom_op(ll, mm, nn-2, [ SS, 1 ], -1, -nn*(nn-1) * gg_mp)
        image_P.update({ (ll+1,mm,nn-1) : nn * gg_mp })

    image_K = {}
    if gg_zp + gg_mz != 0:
        image_K = binom_op(ll+1, mm, nn, 1, mu/4 * (gg_zp + gg_mz))
        del image_K[ll+1,mm,nn]

    image_L = {}
    if gg_zp != 0 and nn != 0:
        if nn >= 2 and ll >= 1:
            factor = -mu*ll*nn*(nn-1)
            image_L = ext_binom_op(ll-1, mm, nn-2, [ SS, 1 ], -1, factor * gg_zp)
        coefficients = [ SS-ll-3/4*(nn-1), -1/2 ]
        image_L.update(insert_z_poly({(ll,mm,nn-1):-1}, coefficients, mu*nn * gg_zp))

    image_M = {}
    if gg_mz != 0 and nn != 0:
        image_M = ext_binom_op(ll, mm, nn-1, [ SS, 1 ], -1, mu*nn * gg_mz)
        coefficients = [ (nn-1)/2, 1 ]
        add_left(image_M, insert_z_poly({(ll,mm,nn-1):1}, coefficients, -mu*nn/2 * gg_mz))

    return sum_vecs([ image_P, image_K, image_L, image_M ])


##########################################################################################
# collective-spin decoherence
##########################################################################################

# diagonal terms of collective-spin decoherence
def op_image_decoherence_diag_collective(op, SS, dec_vec, mu):
    ll, mm, nn = op
    D_p, D_z, D_m = abs(np.array(dec_vec))**2
    if mu == 1:
        D_mu, D_nu = D_p, D_m
    else:
        D_mu, D_nu = D_m, D_p

    image_mu = {}
    if D_mu != 0:
        image_mu = { (ll+1,kk,nn+1) : D_mu * (2**(mm-kk)-1) * binom(mm,kk)
                     for kk in range(mm) }
        coefficients = [ ll*(ll+1) + nn*(nn+1), 2*(ll+nn+1) ]
        image_mu.update(ext_binom_op(*op, coefficients, 1, -D_mu))
        coefficients = [ ll*(ll+1) + nn*(nn+1), 2*(ll+nn+2) ]
        add_left(image_mu, insert_z_poly({op:1}, coefficients, D_mu/2))
        if ll >= 1 and nn >= 1:
            vec = { (ll-1,mm,nn-1) : 1 }
            coefficients = [ (ll-1)*(nn-1), 2*(ll+nn-2), 4 ]
            image_mu.update(insert_z_poly(vec, coefficients, ll*nn * D_mu))

    image_nu = {}
    if D_nu != 0:
        image_nu = binom_op(ll+1, mm, nn+1, 1, -D_nu)
        del image_nu[ll+1,mm,nn+1]
        coefficients = [ ll*(ll-1) + nn*(nn-1), 2*(ll+nn) ]
        image_nu.update(insert_z_poly({op:1}, coefficients, D_nu/2))

    image_z = {}
    if D_z != 0 and ll != nn:
        image_z = { (ll,mm,nn) : -D_z/2 * (ll-nn)**2 }

    return sum_vecs([ image_mu, image_nu, image_z ])

# collective-spin decoherence "Q" cross term
def op_image_decoherence_Q_collective(op, SS, dec_vec, mu):
    ll, mm, nn = op
    g_p, g_z, g_m = dec_vec
    if mu == 1:
        g_mu, g_nu = g_p, g_m
    else:
        g_mu, g_nu = g_m, g_p

    gg_mp = np.conj(g_nu) * g_mu
    gg_zp = np.conj(g_z) * g_mu
    gg_mz = np.conj(g_nu) * g_z
    gg_P = ( gg_zp + gg_mz ) / 2
    gg_M = ( gg_zp - gg_mz ) / 2

    image_P = {}
    if gg_mp != 0:
        image_P = { (ll+2,kk,nn) : -gg_mp * (2**(mm-kk-1)-1) * binom(mm,kk)
                    for kk in range(mm-1) }
        if nn >= 1:
            op_1 = (ll+1, mm, nn-1)
            add_left(image_P, ext_binom_op(*op_1, [ nn, 2 ], 1), nn * gg_mp)
            del image_P[ll+1,mm+1,nn-1]
            add_left(image_P, {op_1 : nn*(-nn+1) * gg_mp})
        if nn >= 2:
            vec = { (ll, mm, nn-2) : -nn*(nn-1) * gg_mp }
            coefficients = [ (nn-1)*(nn-2)/2, (2*nn-3), 2 ]
            add_left(image_P, insert_z_poly(vec, coefficients))

    image_L = {}
    image_M = {}
    if gg_P != 0 or gg_M != 0:
        factor_ll = mu * ( (ll-nn+1/2) * gg_P + (ll+1/2) * gg_M )
        factor_nn = mu * ( (ll-nn+1/2) * gg_P + (nn+1/2) * gg_M )
        image_L = binom_op(ll+1, mm, nn, 1, factor_ll)
        image_L[ll+1,mm,nn] -= factor_nn
        add_left(image_L, ext_binom_op(ll+1, mm, nn, [ 0, 1 ], 1, mu * gg_M))
        del image_L[ll+1,mm+1,nn]

        if nn >= 1:
            factor_mm_0 = (ll-nn+1/2) * gg_P + (ll-1/2) * gg_M
            factor_mm_1 = (ll-nn+1/2) * gg_P + (ll+nn/2-1) * gg_M
            factors = [ -mu*nn*(nn-1) * factor_mm_0,
                        -2*mu*nn * factor_mm_1,
                        -2*mu*nn * gg_M ]
            image_M = { (ll,mm+jj,nn-1) : factors[jj] for jj in range(3) }

    return sum_vecs([ image_P, image_L, image_M ])


##########################################################################################
# image of operators under the time derivative operator
##########################################################################################

# convert decoherence rates and transformation matrix to decoherence vectors
def get_dec_vecs(dec_rates, dec_mat):
    if dec_rates is None: return []
    if dec_rates is not None and len(dec_rates) == 3:
        dec_rates = ( dec_rates, (0,0,0) )
    if dec_mat is None: dec_mat = np.eye(3)
    dec_vecs = []
    for jj in range(3):
        dec_vec_g = dec_mat[:,jj] * np.sqrt(dec_rates[0][jj])
        dec_vec_G = dec_mat[:,jj] * np.sqrt(dec_rates[1][jj])
        if max(abs(dec_vec_g)) == 0 and max(abs(dec_vec_G)) == 0: continue
        dec_vecs.append((dec_vec_g,dec_vec_G))
    return dec_vecs

# compute image of a single operator from decoherence
def op_image_decoherence(op, SS, dec_vec, mu):
    dec_vec_g, dec_vec_G = dec_vec

    image = {}
    image = sum_vecs([ op_image_decoherence_diag_individual(op, SS, dec_vec_g, mu),
                       op_image_decoherence_diag_collective(op, SS, dec_vec_G, mu) ])

    for image_Q, dec_vec in [ ( op_image_decoherence_Q_individual, dec_vec_g ),
                              ( op_image_decoherence_Q_collective, dec_vec_G ) ]:
        Q_lmn = image_Q(op, SS, dec_vec, mu)
        if op[0] == op[2]:
            Q_nml = Q_lmn
        else:
            Q_nml = image_Q(op[::-1], SS, dec_vec, mu)
        add_left(image, Q_lmn)
        add_left(image, conj_vec(Q_nml))

    return image

# compute image of a single operator from coherent evolution
def op_image_coherent(op, h_vec):
    if op == (0,0,0): return {}
    image = {}
    for h_op, h_val in h_vec.items():
        add_left(image, multiply_terms(h_op, op), +1j*h_val)
        add_left(image, multiply_terms(op, h_op), -1j*h_val)
    return image

# full image of a single operator under the time derivative operator
def op_image(op, h_vec, spin_num, dec_vecs, mu):
    image = op_image_coherent(op, h_vec)
    for dec_vec in dec_vecs:
        add_left(image, op_image_decoherence(op, spin_num/2, dec_vec, mu))
    return clean(image)


##########################################################################################
# miscellaneous methods for computing single- and multi-time correlators
##########################################################################################

# list of operators necessary to compute squeezing, specified by (\mu,\z,\nu) exponents
squeezing_ops = [ (0,1,0), (0,2,0), (1,0,0), (2,0,0), (1,1,0), (1,0,1) ]

# get transverse weight of an operator, i.e. max(l,n) for op ~ S_+^l S_z^m S_-^n
def transverse_weight(op_vec):
    return max( max(op[0],op[2]) for op in op_vec.keys() )

##########################################################################################
# collective-spin correlators
##########################################################################################

# compute (factorially suppresed) derivatives of operators,
# returning an operator vector for each derivative order:
# deriv_op_vec[mm,kk] = (d/dt)^kk S_mm / kk!
#                     = \sum_nn T^kk_{mm,nn} S_nn / kk!
def get_deriv_op_vec(order_cap, spin_num, init_state, h_vec,
                     dec_rates = None, dec_mat = None, deriv_ops = squeezing_ops,
                     remove_irrelevant_ops = True, max_step_offset = 0, mu = 1):
    dec_vecs = get_dec_vecs(dec_rates, dec_mat)

    if type(init_state) is str: init_state = axis_str(init_state)
    if remove_irrelevant_ops and tuple(np.sign(init_state)) == (-1,0,0):
        chop_operators = True
        max_transverse_step = max([ op[0] for op in h_vec.keys() ] + [ 0 ])
        pp, zz, mm = 0, 1, 2 # conventional ordering for decoherence vectors
        for vec in dec_vecs:
            if vec[zz][zz] != 0: max_transverse_step = max(max_transverse_step,1)
            if vec[zz][pp] != 0: max_transverse_step = max(max_transverse_step,2)
            if vec[pp][zz] != 0:
                if vec[pp][pp] != 0 or vec[pp][mm] != 0:
                    # through M_{\ell mn}
                    max_transverse_step = max(max_transverse_step,1)
            if vec[pp][pp] != 0:
                if vec[pp][mm] != 0:
                    # through \tilde P_{\ell mn}
                    max_transverse_step = max(max_transverse_step,2)
                else:
                    # through S_\mu jump operator
                    max_transverse_step = max(max_transverse_step,1)
    else:
        chop_operators = False

    diff_op = {} # single time derivative operator
    deriv_op_vec = {} # deriv_op_vec[mm,kk] = (1/kk!) (d/dt)^kk S_mm
                      #                     = (1/kk!) \sum_nn T^kk_{mm,nn} S_nn
    for deriv_op in deriv_ops:
        deriv_op_vec[deriv_op,0] = { deriv_op : 1 }
        for order in range(1,order_cap):
            # compute relevant matrix elements of the time derivative operator
            deriv_op_vec[deriv_op,order] = {}
            for op, val in deriv_op_vec[deriv_op,order-1].items():
                try: add_left(deriv_op_vec[deriv_op,order], diff_op[op], val/order)
                except:
                    diff_op[op] = op_image(op, h_vec, spin_num, dec_vecs, mu)
                    if op[0] != op[-1]:
                        diff_op[op[::-1]] = conj_vec(diff_op[op])
                    add_left(deriv_op_vec[deriv_op,order], diff_op[op], val/order)
            clean(deriv_op_vec[deriv_op,order])

            if chop_operators and order > order_cap // 2:
                # throw out operators with no contribution to correlators
                max_steps = (order_cap-order) * max_transverse_step + max_step_offset
                irrelevant_ops = [ op for op in deriv_op_vec[deriv_op,order].keys()
                                   if op[0] > max_steps or op[2] > max_steps ]
                for op in irrelevant_ops:
                    del deriv_op_vec[deriv_op,order][op]

    return deriv_op_vec

# sandwich a deriv_op_vec object by operators prepend_op and append_op
def sandwich_deriv_op_vec(deriv_op_vec, prepend_op = None, append_op = None):
    new_deriv_op_vec = deriv_op_vec.copy()
    if prepend_op is not None:
        for key in deriv_op_vec.keys():
            new_deriv_op_vec[key] = multiply_vecs(prepend_op,deriv_op_vec[key])
    if append_op is not None:
        for key in deriv_op_vec.keys():
            new_deriv_op_vec[key] = multiply_vecs(deriv_op_vec[key],append_op)
    return new_deriv_op_vec

# add two deriv_op_vecs
def add_deriv_op_vecs(deriv_op_vec_lft, deriv_op_vec_rht, factor_lft = 1, factor_rht = 1):
    ops_lft = set( key[0] for key in deriv_op_vec_lft.keys() )
    ops_rht = set( key[0] for key in deriv_op_vec_rht.keys() )
    order_cap = min( max( key[1] for key in deriv_op_vec_lft.keys() ),
                     max( key[1] for key in deriv_op_vec_rht.keys() ) ) + 1

    ops_both = set(ops_lft).intersection(ops_rht)
    ops_lft_only = set(ops_lft).difference(ops_rht)
    ops_rht_only = set(ops_rht).difference(ops_lft)

    deriv_op_vec_sum = {}
    for order in range(order_cap):
        for op in ops_both:
            deriv_op_vec_sum[op,order] \
                = sum_vecs([ deriv_op_vec_lft[op,order], deriv_op_vec_rht[op,order] ],
                           [ factor_lft, factor_rht ])
        for op in ops_lft_only:
            deriv_op_vec_sum[op,order] \
                = sum_vecs([ deriv_op_vec_lft[op,order] ], [ factor_lft ])
        for op in ops_rht_only:
            deriv_op_vec_sum[op,order] \
                = sum_vecs([ deriv_op_vec_rht[op,order] ], [ factor_rht ])

    for key in deriv_op_vec_sum.keys():
        clean(deriv_op_vec_sum[key])

    return deriv_op_vec_sum

# multiply two deriv_op_vecs
def multiply_deriv_op_vecs(deriv_op_vec_lft, deriv_op_vec_rht,
                           dag_lft = False, dag_rht = False):
    ops_lft = set( key[0] for key in deriv_op_vec_lft.keys() )
    ops_rht = set( key[0] for key in deriv_op_vec_rht.keys() )
    order_cap = min( max( key[1] for key in deriv_op_vec_lft.keys() ),
                     max( key[1] for key in deriv_op_vec_rht.keys() ) ) + 1

    deriv_op_vec_product = {}
    for op_lft, op_rht in itertools.product(ops_lft, ops_rht):
        for order in range(order_cap):
            deriv_op_vec_product[(op_lft,op_rht),order] = {}
            for order_lft in range(order+1):
                order_rht = order - order_lft
                product = multiply_vecs(deriv_op_vec_lft[op_lft,order_lft],
                                        deriv_op_vec_rht[op_rht,order_rht],
                                        dag_lft = dag_lft, dag_rht = dag_rht)
                add_left(deriv_op_vec_product[(op_lft,op_rht),order], product)
            clean(deriv_op_vec_product[(op_lft,op_rht),order])

    return deriv_op_vec_product

# compute (factorially suppresed) derivatives of operators,
# returning a value for each order:
# deriv_vals[op][kk] = < prepend_zxy * [ (d/dt)^kk op ] * append_zxy >_0 / kk!
def deriv_op_vec_to_vals(bare_deriv_op_vec, spin_num, init_state,
                         prepend_op = None, append_op = None, mu = 1):

    deriv_op_vec = sandwich_deriv_op_vec(bare_deriv_op_vec, prepend_op, append_op)

    deriv_vals = {} # deriv_vals[op][kk]
    deriv_ops = set( key[0] for key in deriv_op_vec.keys() )
    order_cap = max( key[1] for key in deriv_op_vec.keys() ) + 1
    for deriv_op in deriv_ops:
        deriv_vals[deriv_op] = np.zeros(order_cap, dtype = complex)

    init_ln_vals = {} # initial values of relevant operators
    init_ln_val_func = init_ln_val_function(spin_num, init_state, mu)
    for deriv_op, order in deriv_op_vec.keys():
        for op, val in deriv_op_vec[deriv_op,order].items():

            # initial magnitude / phase of this operator, assuming we have already computed it
            init_ln_val = init_ln_vals.get(op)

            # if we can't acquire the initial magnitude / phase, compute it
            if init_ln_val is None:
                init_ln_val = init_ln_val_func(op)
                if init_ln_val is None: continue # the initial magnitude is 0
                init_ln_vals[op] = init_ln_val

            term_ln_mag = np.log(abs(val)) + init_ln_val[0]
            term_phase = init_ln_val[1] * np.exp(1j*np.angle(val))
            deriv_vals[deriv_op][order] += term_phase * np.exp(term_ln_mag)

    return deriv_vals

# convert deriv_vals to correlator values at given times
# note: "times" is assumed to be in units of the OAT squeezing strength
def deriv_vals_to_correlators(deriv_vals, times, mu = 1):
    order_cap = list(deriv_vals.values())[0].size
    times_k = np.array([ times**order for order in range(order_cap) ])
    correlators = { op : deriv_vals[op] @ times_k for op in deriv_vals.keys() }
    if mu == 1:
        return correlators
    else:
        # we computed correlators of the form < S_-^ll (-S_z)^mm S_+^nn >,
        #   so we need to invert them into correlators of the form < S_+^ll S_z^mm S_-^nn >
        return invert_vals(correlators)

# compute correlators from evolution under a general Hamiltonian with decoherence
def compute_correlators(times, order_cap, spin_num, init_state, h_vec,
                        dec_rates = None, dec_mat = None, correlator_ops = squeezing_ops,
                        prepend_op = None, append_op = None, mu = 1):
    max_step_offset = sum([ transverse_weight(op) for op in [ prepend_op, append_op ]
                            if op is not None ])
    deriv_op_vec = get_deriv_op_vec(order_cap, spin_num, init_state, h_vec, dec_rates,
                                    dec_mat, correlator_ops, max_step_offset, mu)
    deriv_vals = deriv_op_vec_to_vals(deriv_op_vec, spin_num, init_state,
                                      prepend_op, append_op, mu)
    return deriv_vals_to_correlators(deriv_vals, times, mu)
