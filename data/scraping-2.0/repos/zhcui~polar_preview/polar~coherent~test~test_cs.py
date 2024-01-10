#!/usr/bin/env python

"""
Test coherent state methods.

Authors:
    Zhi-Hao Cui <zhcui0408@gmail.com>
"""

def test_cs_1e():
    import numpy as np
    from pyscf import gto
    from polar.coherent import cshf
    #from polar.lang_firsov import coherent as cshf
    np.set_printoptions(4, linewidth=1000, suppress=True)

    # Hubbard Holstein model
    nao = 4
    nmode = nao
    nelec = 1
    U = 0.0

    hcore = np.zeros((nao, nao))
    for i in range(nao-1):
        hcore[i, i+1] = hcore[i+1, i] = -1.0
    hcore[nao-1, 0] = hcore[0, nao-1] = -1.0  # PBC

    eri = np.zeros((nao, nao, nao, nao))
    eri[range(nao), range(nao), range(nao), range(nao)] = U

    alpha = 1.0
    w_p = 0.5

    w_p_arr = np.zeros((nmode,))
    w_p_arr[:] = w_p

    g = np.sqrt(alpha * w_p)
    h_ep = np.zeros((nmode, nao, nao))
    for x in range(nmode):
        h_ep[x, x, x] = g

    mol = gto.M()
    mol.nao_nr = lambda *args: nao
    mol.nelectron = nelec
    mol.spin = 1
    mol.incore_anyway = True
    mol.verbose = 4
    mol.build(dump_input=False)
    mycs = cshf.UCSHF(mol, h_ep, w_p_arr, zs=None)
    mycs.get_hcore = lambda *args: hcore
    mycs.get_ovlp = lambda *args: np.eye(nao)
    mycs._eri = eri

    e_cshf  = mycs.kernel()
    print ("CSHF energy")
    print (e_cshf)
    assert abs(e_cshf - -2.25000000) < 1e-10

    e_csmp2 = mycs.CSMP2()
    print ("CSMP2 energy")
    print (e_csmp2)
    assert abs(e_cshf + e_csmp2 - -2.37777778) < 1e-7

def test_cs_4e():
    import numpy as np
    from pyscf import gto
    from polar.coherent import cshf
    np.set_printoptions(4, linewidth=1000, suppress=True)

    # Hubbard Holstein model
    nao = 4
    nmode = nao
    nelec = 4
    U = 2.0

    hcore = np.zeros((nao, nao))
    for i in range(nao-1):
        hcore[i, i+1] = hcore[i+1, i] = -1.0
    hcore[nao-1, 0] = hcore[0, nao-1] = -1.0  # PBC

    eri = np.zeros((nao, nao, nao, nao))
    eri[range(nao), range(nao), range(nao), range(nao)] = U

    alpha = 1.0
    w_p = 0.5

    w_p_arr = np.zeros((nmode,))
    w_p_arr[:] = w_p

    g = np.sqrt(alpha * w_p)
    h_ep = np.zeros((nmode, nao, nao))
    for x in range(nmode):
        h_ep[x, x, x] = g

    mol = gto.M()
    mol.nao_nr = lambda *args: nao
    mol.nelectron = nelec
    mol.spin = 0
    mol.incore_anyway = True
    mol.verbose = 4
    mol.build(dump_input=False)
    mycs = cshf.RCSHF(mol, h_ep, w_p_arr, zs=None)
    mycs.get_hcore = lambda *args: hcore
    mycs.get_ovlp = lambda *args: np.eye(nao)
    mycs._eri = eri

    e_cshf  = mycs.kernel()
    print ("CSHF energy")
    print (e_cshf)

    e_csmp2, (t2, rdm1) = mycs.CSMP2(with_t2=True, make_rdm1=True)
    print ("CSMP2 energy")
    print (e_csmp2)
    print ("rdm1")
    print (rdm1)

if __name__ == "__main__":
    test_cs_4e()
    test_cs_1e()
