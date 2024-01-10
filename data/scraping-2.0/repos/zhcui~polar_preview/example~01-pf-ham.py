#!/usr/bin/env python

import numpy as np
from scipy import linalg as la
from pyscf import gto

def get_mol(name, basis, dump_input=False, *args):
    if name == "H2":
        return get_mol_H2(basis, dump_input=dump_input, *args)
    elif name == "HF":
        return get_mol_HF(basis, dump_input=dump_input, *args)
    else:
        raise ValueError

def get_mol_H2(basis, bond_length=0.746, shift=0.0, dump_input=False):
    mol = gto.Mole(atom=[['H', [0, 0, 0.0 - shift]],
                         ['H', [0, 0, bond_length - shift]]],
                   basis=basis,
                   unit='A',
                   verbose=4,
                   )
    mol.build(dump_input=dump_input)
    return mol

def get_mol_HF(basis, bond_length=0.918, shift=0.0, dump_input=False):
    mol = gto.Mole(atom=[['H', [0, 0, 0.0 - shift]],
                         ['F', [0, 0, bond_length - shift]]],
                   basis=basis,
                   unit='A',
                   verbose=4,
                   )
    mol.build(dump_input=dump_input)
    return mol


def solve_mol(mol, w, lc, pol_axis, nmode=1, nph=24):
    """
    Solve molecule with different methods.
    """
    from polar.system import molecule as mole
    from polar.lang_firsov import lang_firsov as lf
    from pyscf import gto, scf, ao2mo
    from pyscf import mp, cc
    from pyscf import lo

    nao = mol.nao_nr()
    nelec = mol.nelectron

    mol_fake, H0, hcore, ovlp_lo, eri, w_p, h_ep, rdm1_lo = \
            mole.get_pf_ham(mol, w, lc, pol_axis, nmode=nmode)

    mf_lo = scf.RHF(mol_fake)
    mf_lo.energy_nuc = lambda *args: H0
    mf_lo.get_hcore = lambda *args: hcore
    mf_lo.get_ovlp = lambda *args: ovlp_lo
    mf_lo._eri = eri
    mf_lo.kernel(dm0=rdm1_lo)

    e_elec_hf = mf_lo.e_tot
    rdm1_lo_re = mf_lo.make_rdm1()
    print ("HF energy (elec)", e_elec_hf)
    print ("rdm1 in LO basis")
    print (rdm1_lo)
    print ()
    print ("-" * 79)
    print ()

    mymp = mf_lo.MP2()
    e_corr_mp2_1, t2 = mymp.kernel()
    e_elec_mp2_1 = mymp.e_tot
    print ("MP2 energy (elec)", e_elec_mp2_1)
    print ()
    print ("-" * 79)
    print ()

    mycc = mf_lo.CCSD()
    mycc.max_cycle = 200
    e_corr_cc_1, t1, t2 = mycc.kernel()
    e_elec_cc_1 = mycc.e_tot

    dE_ccsd_mp2_1 = e_elec_cc_1 - e_elec_mp2_1
    print ("CCSD energy (elec)", e_elec_cc_1)
    print ()
    print ("-" * 79)
    print ()

    from polar.lang_firsov import coherent as coh
    zs = coh.get_zs(h_ep, w_p, rdm1_lo)
    mf_cshf = coh.RCSHF(mol_fake, h_ep=h_ep, w_p=w_p, zs=zs)
    mf_cshf.energy_nuc = lambda *args: H0
    mf_cshf.get_hcore = lambda *args: hcore
    mf_cshf.get_ovlp = lambda *args: ovlp_lo
    mf_cshf._eri = eri
    mf_cshf.kernel(dm0=rdm1_lo)
    rdm1 = mf_cshf.make_rdm1()
    zs = mf_cshf.zs

    e_cshf = mf_cshf.e_tot
    print ("CS-HF energy", e_cshf)
    print ()
    print ("-" * 79)
    print ()

    #e_coh = coh.get_e_coh(mf_cshf, h_ep, w_p)
    e_csmp2, _ = coh.get_e_csmp2(mf_cshf, h_ep, w_p, zs=zs)
    e_csmp2 += e_cshf
    print ("CS-MP2 energy", e_csmp2)
    print ()
    print ("-" * 79)
    print ()

    mymp = mf_cshf.MP2()
    e_corr_mp2_2, t2 = mymp.kernel()
    e_elec_mp2_2 = mymp.e_tot
    print ("MP2 energy (based on CS-HF)", e_elec_mp2_2)
    print ()
    print ("-" * 79)
    print ()

    mycc = mf_cshf.CCSD()
    mycc.max_cycle = 200
    e_corr_cc_2, t1, t2 = mycc.kernel()
    e_elec_cc_2 = mycc.e_tot
    e_t = mycc.ccsd_t()
    print ("(T) correction for CSMP2", e_t)

    dE_ccsd_mp2_2 = e_elec_cc_2 - e_elec_mp2_2 + e_t
    print ("CCSD energy (based on CS-HF)", e_elec_cc_2)
    print ()
    print ("-" * 79)
    print ()

    e_csmp2_corr = e_csmp2 + dE_ccsd_mp2_2
    print ("CS-MP2 energy (w. corr.)", e_csmp2_corr)
    print ()
    print ("-" * 79)
    print ()

    # FCI
    print ("-" * 79)
    if nao * nelec <= 100:
        from polar.fci import fci
        e_fci, civec = fci.kernel(hcore, eri, nao, nelec, nmode, nph, h_ep, w_p,
                              shift_vac=True, ecore=H0, tol=1e-8, max_cycle=10000)
        print ("ED", e_fci)
    else:
        e_fci = -1
    print ()
    print ("-" * 79)
    print ()

    mo_coeff = mf_cshf.mo_coeff
    mo_occ = mf_cshf.mo_occ

    mylf = lf.GLangFirsov(mol=mol_fake, h0=H0, h1=hcore, h2=eri, ovlp=ovlp_lo,
                          h_ep=h_ep, w_p=w_p,
                          nelec=nelec, uniform=False)

    params = (np.random.random(mylf.nparam_full) - 0.5) #* 0.0
    params[:] = 0.0
    params[-nmode:] = zs
    e = mylf.kernel(params=params,
                    use_num_grad=False, full_opt=True,
                    mo_coeff=mo_coeff, mo_occ=mo_occ, conv_tol=1e-8,
                    max_cycle=1000, ntrial=5, mp2=True, nph=17)

    e_lfmp2 = mylf.e_tot
    e_lfhf = e_lfmp2 - mylf.e_mp2

    print ("LF-HF energy", e_lfhf)
    print ()
    print ("-" * 79)
    print ()

    print ("LF-MP2 energy", e_lfmp2)
    print ()
    print ("-" * 79)
    print ()

    e_lfmp2_corr = e_lfmp2 + dE_ccsd_mp2_2
    print ("LF-MP2 energy (w. corr.)", e_lfmp2_corr)
    print ()
    print ("-" * 79)
    print ()

    mylf._scf._eri = mylf._scf._eri.reshape(-1)
    from pyscf.cc import gccsd
    mycc = gccsd.GCCSD(mylf._scf)
    mycc.level_shift = 1e-8
    #mycc = cc.GCCSD(mylf._scf)
    mycc.max_cycle = 200
    e_mp2_x = mycc.init_amps()[0]
    e_cc_x = mycc.kernel()[0]
    e_cc_t_x = mycc.ccsd_t()

    print ("(T) correction for LFMP2", e_cc_t_x)
    e_corr = e_cc_x + e_cc_t_x - e_mp2_x

    e_test = e_lfmp2 + e_corr
    print ("LF-MP2 energy (w. corr. test)", e_test)
    print ()
    print ("-" * 79)
    print ()

    return e_elec_hf, e_elec_mp2_1, e_elec_cc_1, e_cshf, e_csmp2, e_csmp2_corr,\
           e_lfhf, e_lfmp2, e_lfmp2_corr, e_fci, e_test

if __name__ == "__main__":
    np.set_printoptions(3, linewidth=1000, suppress=True)
    np.random.seed(10086)

    nmode = 1
    settings = [{"name": "H2", "basis": '631g', "w": 0.466751, "lc": 0.05, "pol_axis": 2, "nmode": nmode},
                #{"name": "H2", "basis": '6311++g**', "w": 0.466751, "lc": 0.05, "pol_axis": 2, "nmode": nmode},
                {"name": "HF", "basis": 'sto6g', "w": 0.531916, "lc": 0.05, "pol_axis": 2, "nmode": nmode},
                #{"name": "HF", "basis": '6311++g**', "w": 0.531916, "lc": 0.05, "pol_axis": 2, "nmode": nmode}
                ]
    for sets in settings:
        name = sets["name"]
        basis = sets["basis"]
        mol = get_mol(name, basis)
        w = sets["w"]
        lc = sets["lc"]
        pol_axis = sets["pol_axis"]
        nmode = sets["nmode"]

        e_elec_hf, e_elec_mp2, e_elec_cc, e_cshf, e_csmp2, e_csmp2_corr,\
                e_lfhf, e_lfmp2, e_lfmp2_corr, e_fci, e_test\
                = solve_mol(mol, w, lc, pol_axis, nmode=nmode, nph=24)
        print ("-" * 79)
        for key, val in sets.items():
            print ("%30s : %15s"%(key, val))
        print ("-" * 79)
        print ("%30s %15.8f"%("Eelec (HF)", e_elec_hf))
        print ("%30s %15.8f"%("Eelec (MP2)", e_elec_mp2))
        print ("%30s %15.8f"%("Eelec (CCSD)", e_elec_cc))

        print ("%30s %15.8f"%("E (CS-HF)", e_cshf))
        print ("%30s %15.8f"%("E (CS-MP2)", e_csmp2))
        print ("%30s %15.8f"%("E (CS-MP2 + corr.)", e_csmp2_corr))

        print ("%30s %15.8f"%("E (LF-HF)", e_lfhf))
        print ("%30s %15.8f"%("E (LF-MP2)", e_lfmp2))
        #print ("%30s %15.8f"%("E (LF-MP2) + corr.", e_lfmp2_corr))
        print ("%30s %15.8f"%("E (LF-MP2 + corr.)", e_test))

        print ("%30s %15.8f"%("E (ED)", e_fci))
        print ("-" * 79)

