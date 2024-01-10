#!/usr/bin/env python3

import os, sys, scipy
import numpy as np

from time import time as system_time

from dicke_methods import coherent_spin_state
from correlator_methods import compute_deriv_vals, dec_mat_drive, \
    mat_zxy_to_pzm, vec_zxy_to_pzm

start_time = system_time()

if len(sys.argv[1:]) not in  [ 3, 4 ]:
    print(f"usage: {sys.argv[0]} method lattice_depth lattice_size [rational]")
    exit()

method = sys.argv[1]
lattice_depth = sys.argv[2]
lattice_size = int(sys.argv[3])
rational_correlators = ( len(sys.argv[1:]) == 4 )

TAT, TNT = "TAT", "TNT"
assert(method in [ TAT, TNT ])

data_dir = os.path.dirname(os.path.realpath(__file__)) + "/../data/"
output_dir = data_dir + "trunc/"
file_name = "_".join(sys.argv[1:]) + ".txt"

lattice_dim = 2
confining_depth = 60 # recoil energies
dec_time_SI = 10 # seconds
order_cap = 70

recoil_energy_NU = 21801.397815091557
drive_mod_index_zy = 0.9057195866712102 # for TAT protocol about (z,y)
drive_mod_index_xy_1 = 1.6262104442160061 # for TAT protocol about (x,y)
drive_mod_index_xy_2 = 2.2213461342426544 # for TAT protocol about (x,y)

spin_num = lattice_size**lattice_dim

h_TAT_zxy = { (0,2,0) : +1/3,
              (0,0,2) : -1/3 }
h_TNT_zxy = { (0,2,0) : 1,
              (1,0,0) : -spin_num/2 }

def get_val_1D(depth, file_name):
    file_path = data_dir + file_name
    if not os.path.isfile(file_path):
        print(f"cannot find data file: {file_path}")
        exit()
    with open(file_path, "r") as f:
        for line in f:
            if line[0] == "#": continue
            if line.split(",")[0][:len(depth)] == depth:
                return float(line.split(",")[1])

def get_val_2D(depth, confinement, file_name):
    file_path = data_dir + file_name
    if not os.path.isfile(file_path):
        print(f"cannot find data file: {file_path}")
        exit()
    conf_idx = None
    with open(file_path, "r") as f:
        for line in f:
            if line[0] == "#": continue
            if conf_idx == None:
                conf_idx = [ int(x) for x in line.split(",")[1:] ].index(confinement) + 1
                continue
            if line.split(",")[0][:len(depth)] == depth:
                return float(line.split(",")[conf_idx])

J = get_val_1D(lattice_depth, "J_0.txt")
U = get_val_2D(lattice_depth, confining_depth, f"U_int_{lattice_dim}D.txt")
phi = get_val_2D(lattice_depth, confining_depth, f"phi_opt_{lattice_dim}D.txt")

if None in [ J, U, phi ]:
    print("could not find values for J, U, or phi... you should inquire")
    print(f"J: {J}")
    print(f"U: {U}")
    print(f"phi: {phi}")
    exit()

h_std = 2**(1+lattice_dim/2)*J*np.sin(phi/2)
chi = h_std**2 / U / (spin_num-1)

dec_rate_LU = 1/dec_time_SI / recoil_energy_NU
dec_rate = dec_rate_LU / chi
dec_rates = [ (0, dec_rate, dec_rate), (0, 0, 0) ]

init_state = "-Z"
basis_change_zxy = np.array([ [ 0, -1, 0 ],
                              [ 1,  0, 0 ],
                              [ 0,  0, 1 ]])
basis_change = mat_zxy_to_pzm(basis_change_zxy)

if method == TNT:
    h_vec_zxy = h_TNT_zxy
    dec_mat = basis_change
else: # method == TAT
    h_vec_zxy = h_TAT_zxy
    dec_mat = dec_mat_drive(scipy.special.jv(0,drive_mod_index_zy)) @ basis_change
h_vec = vec_zxy_to_pzm(h_vec_zxy)

header = f"# lattice_dim: {lattice_dim}\n"
header += f"# confining depth (E_R): {confining_depth}\n"
header += f"# order_cap: {order_cap}\n"

op_vals = compute_deriv_vals(order_cap, spin_num, init_state, h_vec, dec_rates, dec_mat)

if not os.path.isdir(output_dir): os.mkdir(output_dir)

with open(output_dir + file_name, "w") as f:
    f.write(header)
    f.write("# operators: " + " ".join([ str(op) for op, _ in op_vals.items() ]) + "\n")
    for _, vals in op_vals.items():
        f.write(" ".join([ str(val) for val in vals ]) + "\n")

print(f"runtime (seconds): {system_time()-start_time}")
