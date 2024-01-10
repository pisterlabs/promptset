import qutip
import numpy as np
from numpy import pi
from algo.coherentcontrol import CoherentControl, Operation
from algo.metrics import fidelity
from typing import Callable


def get_gkp_cost_function(num_atoms:int, form="square") -> Callable[[np.matrix], float]:
    """get_gkp_cost_function

    Args:
        num_atoms (int)
        form (str): "square"/"hex"

    Returns:
        Callable[[np.matrix], float]
    """
    target = gkp_state(num_atoms, form)

    def cost_func(rho:np.matrix)->float:
        return -1*fidelity(rho, target)

    return cost_func



def gkp_state(num_atoms:int, form="square")->np.matrix:
    """gkp_state Create a GKP state density matrix.

    Args:
        num_moments (int)
        form (str): "square"/"hex"

    Returns:
        rho: Density-Matrix
    """
    # Get Ket Staet:
    if form=="square":
        psi = _goal_gkp_state_ket_square(num_atoms)
    elif form =="hex":
        psi = _goal_gkp_state_ket_alternative(num_atoms, form)
    else:
        raise ValueError(f"Not a possible option: form={form}")
    # Transform to DM:
    rho = qutip.ket2dm(psi).full()
    return rho
    

def _goal_gkp_state_ket_alternative(
    num_moments:int,
    form:str,  # square\hex
    d_b = 10.0
):

    # Derive sequence params:
    if form=="square":
        alpha1 = 1j * np.sqrt( pi/8 )
        alpha2 = np.sqrt( pi/2 )
        times1 = 4
        times2 = 1
    elif form=="hex":
        alpha1 = np.sqrt( pi / (4*np.sqrt(3)) )
        alpha2 = alpha1 * np.exp( 2*1j*pi/3 )
        times1 = 2
        times2 = 2
    else:
        raise ValueError(f"Not a possible option: form={form}")

    n = num_moments + 1 
    r = np.log( np.sqrt( d_b ) )
    m = round( (np.exp(r)**2) /pi )  # 3 when dB=10
    D = lambda alpha: qutip.displace(n, alpha)

    # Perform sequence:
    psi = qutip.basis(n, 0)
    for _ in range(m*times1):
        psi = ( D(alpha1)+D(-alpha1) ) * psi    
    for _ in range(m*times2):
        psi = ( D(alpha2)+D(-alpha2) ) * psi    

    # Normalize:
    psi = psi.unit()
    return psi

    
def _goal_gkp_state_ket_square(
    num_moments:int,
    d_b = 10.0
):
    # Constants:
    alpha =  np.sqrt(pi/2)
    
    # Derivedd from d_b:
    r = np.log( np.sqrt( d_b ) )
    m = round( (np.exp(r)**2) /pi )  # 3 when dB=10
    
    # Basic operators:
    n = num_moments + 1 
    S = qutip.squeeze(n, r)
    D_alpha = qutip.displace(n, alpha) + qutip.displace(n, -alpha)
    
    # Perform sequence:
    psi = qutip.basis(n, 0)
    psi = S * psi
    for _ in range(m):
        psi = D_alpha * psi    

    # Normalize:
    psi = psi.unit()

    return psi

def _test_plot_goal_gkp():
    from utils.visuals import plot_matter_state, plot_plane_wigner

    num_moments = 40
    for form in ["square", "hex"]:
        rho = gkp_state(num_moments, form)
        plot_plane_wigner(rho)
    plot_matter_state(rho)
    
    print("Done.")

if __name__ == '__main__':
    _test_plot_goal_gkp()
    print("Done.")

