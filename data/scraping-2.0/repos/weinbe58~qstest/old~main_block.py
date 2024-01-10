import sys,os
qspin_path = os.path.join(os.getcwd(),"../QuSpin_dev/")
sys.path.insert(0,qspin_path)

from quspin.basis import spin_basis_1d,photon_basis # Hilbert space bases
from quspin.operators import hamiltonian,exp_op # Hamiltonian and observables
from quspin.tools.measurements import obs_vs_time # t_dep measurements
from quspin.tools.Floquet import Floquet,Floquet_t_vec # Floquet Hamiltonian
from quspin.basis.photon import coherent_state # HO coherent state
from block_evo import block_expm
from itertools import izip
import numpy as np # generic math functions
import scipy.sparse.linalg as sp_sla
import sys,os
from memory_profiler import profile



def drive(t,Omega):
	return np.sin(Omega*t)

#@profile
def main(Nph_max,Nph,L,Omega,nT,nT0,len_T,blocks,n_jobs=1):
	Jzz=1.0
	hz=0.809
	hx=0.9045
	A=hz # spin-photon coupling strength (drive amplitude) 
	#
	##### set up photon-atom Hamiltonian #####
	# define operator site-coupling lists
	ph_energy=[[Omega]] # photon energy

	absorb=[[A/(2.0*np.sqrt(Nph)),i] for i in range(L)] # absorption term	
	emit=[[A/(2.0*np.sqrt(Nph)),i] for i in range(L)] # emission term

	z_field=[[hz,i] for i in range(L)] # atom energy
	x_field=[[hx,i] for i in range(L)] # atom energy
	J_nn=[[Jzz,i,(i+1)%L] for i in range(L)] # atom energy

	# define static and dynamics lists
	static=[["|n",ph_energy],["x|-",absorb],["x|+",emit],["z|",z_field],["x|",x_field],["zz|",J_nn]]
	static_sp=[["z|",z_field],["x|",x_field],["zz|",J_nn]]
	# compute atom-photon basis
	#basis=photon_basis(spin_basis_1d,L=L,Nph=Nph_max)
	basis=photon_basis(spin_basis_1d,L,Nph=Nph_max,kblock=0,pblock=1)
	basis_full=photon_basis(spin_basis_1d,L,Nph=Nph_max)
	basis_sc=spin_basis_1d(L,kblock=0,pblock=1)
#	basis_sc_full = spin_basis_1d(L)
	P = basis.get_proj(np.float64)
	P_sc = basis_sc.get_proj(np.float64)


	# compute atom-photon Hamiltonian H
	H = hamiltonian(static,[],dtype=np.float64,basis=basis)
	H_sp = hamiltonian(static_sp,[],dtype=np.float64,basis=basis)
	print("ph-full H-space dim", basis_full.Ns)
	print("ph-block H-space dim", basis.Ns)
	print("spin-block H-space dim", basis_sc.Ns)

	#### define observables #####
	# in atom-photon Hilbert space
	ph_args={"basis":basis,"check_symm":False,"check_herm":False,"check_pcon":False,"dtype":np.float64}
	ph_args_full={"basis":basis_full,"check_symm":False,"check_herm":False,"check_pcon":False,"dtype":np.float64}
	n =hamiltonian([["|n",[[1.0, ]] ]],[],**ph_args)
	sz_ph=hamiltonian([["z|",[[1.0,0]] ]],[],**ph_args_full)

	#
	static_sc=[["z",z_field],["x",x_field],["zz",J_nn]]
	dynamic_sc=[["x",x_field,drive,[Omega]]]
	H_sc=hamiltonian(static_sc,dynamic_sc,dtype=np.float64,basis=basis_sc) 
	sc_args={"basis":basis_sc,"check_symm":False,"check_herm":False,"check_pcon":False,"dtype":np.float64}
	
	# define sc observables below
	sz_sc=hamiltonian([["z",[[1.0,0]] ]],[],**sc_args)


	##### define initial state #####
	# define atom ground state
	E_sc,V_sc=H_sc.eigsh(k=2,time=0,which='BE',maxiter=1E4)
	W = np.diff(E_sc).squeeze()
	psi_sp_i=V_sc[:,0].ravel()
	print("spin MB bandwidth is %s" %(W) )

	# define photon Flock state containing Nph_max photons
	#psi_ph_i=np.zeros((Nph_max+1,),dtype=np.float64);psi_ph_i[Nph] = 1.0
	psi_ph_i = coherent_state(np.sqrt(Nph),Nph_max+1)
	#print np.linalg.norm(psi_ph_i)
	psi_ph_i /= np.linalg.norm(psi_ph_i)
	# compute atom-photon initial state as a tensor product
	psi_sp_ph_i=np.kron(psi_sp_i,psi_ph_i)
	#
	##### calculate time evolution #####
	# define time vector over 100 driving cycles with 100 points per period
	t=Floquet_t_vec(Omega,nT,len_T=len_T) # t.i = initial time, t.T = driving period 
	t0=nT0*t.T # time from which correlator is measured
	T = t.T
	# evolve atom-photon state up to time t0

	psi_ph_t0 = exp_op(H,a=-1j*t0).dot(psi_sp_ph_i) # spin photon state

	psi_n = n.dot(psi_ph_t0)
	psi_ph_t0_full = P.dot(psi_ph_t0)
	psi_sz_t0_full = sz_ph.dot(psi_ph_t0_full)

	O_n_t0 = n.matrix_ele(psi_ph_t0,psi_ph_t0)
	O_sz_ph_t0 = sz_ph.matrix_ele(psi_ph_t0_full,psi_ph_t0_full)

	expH = exp_op(H,a=-1j,start=0,stop=nT*T,num=len_T*nT+1,endpoint=True,iterate=True)
	psi_ph_t = expH.dot(psi_ph_t0)
	psi_n_t = expH.dot(psi_n)
	psi_sz_t = block_expm(blocks,static,np.float64,photon_basis,(spin_basis_1d,L),psi_sz_t0_full,0,nT*T,num=len_T*nT+1,endpoint=True,iterate=True,n_jobs=n_jobs)

	# semi-classical
	evo_dict={'H':H_sc,'T':T}
	HF=Floquet(evo_dict,n_jobs=2,HF=True).HF

	psi_sp_t0_sc = exp_op(HF,a=-1j*t0).dot(psi_sp_i) # spin state, driven

	psi_sp_t0_sc = sp_sla.expm_multiply(-1j*t0*HF,psi_sp_i)
	
	exit()

	psi_ph_t0_full = P_sc.dot(psi_ph_t0_sc)
	psi_sz_t0_full_sc = sz_sc.dot(psi_sz_t0_full_sc)
	O_sz_sc_t0 = sz_sc.matrix_ele(psi_sz_t0_full_sc,psi_sz_t0_full_sc)
	
	exit()

	O_n = np.zeros((len(t),),dtype=np.float64)
	E_ph = np.zeros_like(O_n)
	nn = np.zeros((len(t),),dtype=np.complex128)
	SzSz_ph = np.zeros((len(t),),dtype=np.complex128)
	O_sz_ph = np.zeros_like(SzSz_ph,dtype=np.float64)

	for i,psi,psi_n,psi_sz in izip(range(len(t)),psi_ph_t,psi_n_t,psi_sz_t):
		print t0+t[i]
		O_n[i] = n.matrix_ele(psi,psi).real
		psi_full = P.dot(psi)
		O_sz_ph[i] = sz_ph.matrix_ele(psi_full,psi_full).real
		E_ph[i] = H_sp.matrix_ele(psi,psi).real/L

		SzSz_ph[i] = sz_ph.matrix_ele(psi_full,psi_sz)
		nn[i] = n.matrix_ele(psi,psi_n)



	SzSz_ph -= O_sz_ph*O_sz_ph_t0
	nn -= O_n*O_n_t0


	##### plot results #####
	import matplotlib.pyplot as plt
	import pylab
	# define legend labels
	str_n = "$\\langle n\\rangle/n_0,$"
	str_z = "$\\langle\\sigma^z\\rangle,$"
	str_zz_Re = "$\\frac{1}{2}\\langle\\{\\sigma^z(t+t_0),\\sigma^z(t_0)\\}\\rangle,$"
	str_zz_Im = "$\\langle[\\sigma^z(t+t_0),\\sigma^z(t_0)]\\rangle,$"
	str_nn_Re = "$\\frac{1}{2}\\langle\\{n(t+t_0),n(t_0)\\}\\rangle,$"
	str_nn_Im = "$\\langle[n(t+t_0),n(t_0)]\\rangle]$"
	str_e = "$\\epsilon(t)$"
	# plot spin-photon data
	fig = plt.figure()
	plt.plot( (t0+t.vals)/t.T,O_n/(Nph),"k",linewidth=1,label=str_n)
	plt.plot( (t0+t.vals)/t.T,O_sz_ph,"b",linewidth=1,label=str_z)
	plt.plot( (t0+t.vals)/t.T,SzSz_ph.real,"m",linewidth=1,label=str_zz_Re)
	#plt.plot( (t0+t.vals)/t.T,E_ph,"r",linewidth=1,label=str_e)
	#plt.plot( (t0+t.vals)/t.T,2.0*SzSz.imag,"--m",linewidth=1,label=str_zz_Im)
	#plt.plot( (t0+t.vals)/t.T,nn.real,"c",linewidth=1,label=str_nn_Re)
	#plt.plot( (t0+t.vals)/t.T,nn.imag,"--c",linewidth=1,label=str_zz_Re)
	# label axes
	plt.xlabel("$t/T$",fontsize=18)
	plt.legend(loc="best") 
	plt.tick_params(labelsize=16)
	plt.grid(True)
	plt.title("Quantum")
	plt.show() 

if __name__ == '__main__':
	#
	##### define model parameters #####
	Nph_max=30 # maximum photon occupation 
	Nph=18
	L=4
	nT=50
	nT0=50
	len_T = 10
	Omega=20.0 # drive frequency

	n_jobs = L+1 # of processes spawned for parallel evo; ideally n_jobs+1 is divisor of # blocks

	os.environ['OMP_NUM_THREADS'] = str(n_jobs) # threads for the job; needs to match job script -pe omp
	# numerical ops do not use more than one core 
	os.environ['NUMEXPR_NUM_THREADS'] = '1'
	os.environ['MKL_NUM_THREADS'] = '1' #

	blocks=[]
	for kblock in range(L/2+1):
		for pblock in [-1,1]:
			blocks.append({"kblock":kblock,"pblock":pblock,"Nph":Nph_max})

	main(Nph_max,Nph,L,Omega,nT,nT0,len_T,blocks,n_jobs=n_jobs)


