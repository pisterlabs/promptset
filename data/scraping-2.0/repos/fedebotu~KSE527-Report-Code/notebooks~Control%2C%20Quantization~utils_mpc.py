import torch
from torch import cos, sin, sign
# from .template import ControlledSystemTemplate
import torch
import torch.nn as nn

class ControlledSystemTemplate(nn.Module):
    """Template Model compatible with hypersolvers
    The hypersolver is defined inside of the dynamics module
    """
    def __init__(self, u, 
                 solver='euler', 
                 hypersolve=None, 
                 retain_u=False, 
                 use_torchdyn=True,
                 multistage=False,
                 _use_xfu=False):
        super().__init__()
        self.u = u
        self.solver = solver
        self.hypersolve = hypersolve
        self.retain_u = retain_u # use for retaining control input (e.g. MPC simulation)
        self.nfe = 0 # count number of function evaluations of the vector field
        self.cur_f = None # current dynamics evaluation
        self.cur_u = None # current controller value
        self._retain_flag = False # temporary flag for evaluating the controller only the first time
        if use_torchdyn: from torchdyn.numerics.odeint import odeint
        else: from torchdiffeq import odeint
        self.odeint = odeint
        self.use_torchdyn = use_torchdyn
        self.multistage = multistage
        self._use_xfu = _use_xfu

    def forward(self, x0, t_span):
        x = [x0[None]]
        xt = x0
        if self.hypersolve:
            # Use the hypersolver to carry forward the system simulation
            for i in range(len(t_span)-1):
                '''HyperEuler step
                x(t+1) = x(t) + Δt*f + Δt^2*g(x,f,u)'''
                Δt = t_span[i+1] - t_span[i]
                f = self._dynamics(t_span[i], xt)
                if self._use_xfu:
                    xfu = torch.cat([xt, self.cur_f, self.cur_u], -1)
                    g = self.hypersolve(0., xfu)
                else:
                    g = self.hypersolve(0., xt)
                self._retain_flag = False
                if self.multistage:
                    half_state_dim = g.shape[-1] // 2
                    g1, g2 = g[..., :half_state_dim], g[..., half_state_dim:]
                    xt = xt + Δt* (f + g1) + (Δt**2)*g2
                else:
                    xt = xt + Δt*f + (Δt**2)*g
                x.append(xt[None])
            traj = torch.cat(x)
        elif self.retain_u:
            '''Iterate over the t_span: evaluate the controller the first time only and then retain it'''
            for i in range(len(t_span)-1):
                self._retain_flag = False
                diff_span = torch.linspace(t_span[i], t_span[i+1], 2)
                if self.use_torchdyn: xt = self.odeint(self._dynamics, xt, diff_span, solver=self.solver)[1][-1]
                else: xt = self.odeint(self._dynamics, xt, diff_span, method=self.solver)[-1]
                x.append(xt[None])
            traj = torch.cat(x)
        else:
            '''Compute trajectory with odeint and base solvers'''
            if self.use_torchdyn: traj = self.odeint(self._dynamics, xt, t_span, solver=self.solver)[1][None]
            else: traj = self.odeint(self._dynamics, xt, t_span, method=self.solver)[None]
            # traj = odeint(self._dynamics, xt, t_span, method=self.solver)[None]
        return traj

    def reset_nfe(self):
        """Return number of function evaluation and reset"""
        cur_nfe = self.nfe; self.nfe = 0
        return cur_nfe

    def _evaluate_controller(self, t, x):
        '''
        If we wish not to re-evaluate the control input, we set the retain
        flag to True so we do not re-evaluate next time
        '''
        if self.retain_u:
            if not self._retain_flag:
                self.cur_u = self.u(t, x)
                self._retain_flag = True
            else: 
                pass # We do not re-evaluate the control input
        else:
            self.cur_u = self.u(t, x)
        return self.cur_u
    
        
    def _dynamics(self, t, x):
        '''
        Model dynamics in the form xdot = f(t, x, u)
        '''
        raise NotImplementedError


class CartPoleGymVersion(ControlledSystemTemplate):
    '''Continuous version of the OpenAI Gym cartpole
    Inspired by: https://gist.github.com/iandanforth/e3ffb67cf3623153e968f2afdfb01dc8'''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)        
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5
        self.polemass_length = (self.masspole * self.length)
        
    def _dynamics(self, t, x_):
        self.nfe += 1 # increment number of function evaluations
        u = self._evaluate_controller(t, x_) # controller
        
        # States
        x   = x_[..., 0:1]
        dx  = x_[..., 1:2]
        θ   = x_[..., 2:3]
        dθ  = x_[..., 3:4]
        
        # Auxiliary variables
        cosθ, sinθ = cos(θ), sin(θ)
        temp = (u + self.polemass_length * dθ**2 * sinθ) / self.total_mass
        
        # Differential Equations
        ddθ = (self.gravity * sinθ - cosθ * temp) / \
                (self.length * (4.0/3.0 - self.masspole * cosθ**2 / self.total_mass))
        ddx = temp - self.polemass_length * ddθ * cosθ / self.total_mass
        self.cur_f = torch.cat([dx, ddx, dθ, ddθ], -1)
        return self.cur_f

    def render(self):
        raise NotImplementedError("TODO: add the rendering from OpenAI Gym")


class CartPole(ControlledSystemTemplate):
    """
    Realistic, continuous version of a cart and pole system. This version considers friction for the cart and the pole. 
    We do not consider the case in which the normal force can be negative: reasonably, the cart should not try to "jump off" the track. 
    This also allows us not needing to consider the previous step's sign.
    References: 
        - http://coneural.org/florian/papers/05_cart_pole.pdf
        - https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-832-underactuated-robotics-spring-2009/readings/MIT6_832s09_read_ch03.pdf
        - https://gist.github.com/iandanforth/e3ffb67cf3623153e968f2afdfb01dc8
        - https://github.com/AadityaPatanjali/gym-cartpolemod
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)        
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5
        self.polemass_length = (self.masspole * self.length)
        self.frictioncart = 0 # 5e-4
        self.frictionpole = 0 # 2e-6

    def _dynamics(self, t, x_):
        self.nfe += 1 # increment number of function evaluations
        u = self._evaluate_controller(t, x_) # controller
        
        # States
        x, dx, θ, dθ = self._divide_states(x_)
        
        # Auxiliary variables
        cosθ, sinθ = cos(θ), sin(θ)
        temp = (u + self.polemass_length * dθ**2 * sinθ) / self.total_mass 
        signed_μc = self.frictioncart * sign(dx)
        μp = self.frictionpole

        # Differential Equations
        nom_ddθ = self.gravity * sinθ - (μp * dθ) / (self.masspole * self.length) - \
                         cosθ * (temp + (self.masspole * self.length * dθ**2 * signed_μc * cosθ) / self.total_mass - signed_μc * self.gravity) # nominator ddθ
        den_ddθ = self.length * (4/3 - self.masspole * cosθ * (cosθ - signed_μc) / self.total_mass) # denominator ddθ
        ddθ = nom_ddθ / den_ddθ # angular acceleration of the pole
        nc = (self.masscart + self.masspole) * self.gravity - self.masspole * self.length * (ddθ * sinθ + dθ**2 * cosθ) # normal force cart
        ddx = temp + (- self.polemass_length * ddθ * cosθ - signed_μc * nc) / self.total_mass # acceleration of the track
        self.cur_f = torch.cat([dx, ddx, dθ, ddθ], -1)
        return self.cur_f

    def _divide_states(self, x_):
        x   = x_[..., 0:1]
        dx  = x_[..., 1:2]
        θ   = x_[..., 2:3]
        dθ  = x_[..., 3:4]
        return x, dx, θ, dθ

    def kinetic_energy(self, x_):
        x, dx, θ, dθ = self._divide_states(x_) 
        return 1/2 * (self.masscart + self.masspole) * dx**2 + self.masspole * dx * dθ * self.length * cos(θ) + 1/2 * self.masspole * self.length**2 * dθ**2

    def potential_energy(self, x_):
        x, _, θ, _ = self._divide_states(x_) 
        return self.masspole * self.gravity * self.length * cos(θ)

    def dynamics(self, t, x):
        return self._dynamics(t, x) # compatibility

    def render(self):
        raise NotImplementedError("TODO: add the rendering from OpenAI Gym")


import torch
from torch import mm
from torch import nn
from warnings import warn
tanh = nn.Tanh() 

class BoxConstrainedController(nn.Module):
    """Simple controller  based on a Neural Network with
    bounded control inputs

    Args:
        in_dim: input dimension
        out_dim: output dimension
        hid_dim: hidden dimension
        zero_init: initialize last layer to zeros
    """
    def __init__(self, 
                 in_dim, 
                 out_dim, 
                 h_dim=64, 
                 num_layers=2, 
                 zero_init=True,
                 input_scaling=None, 
                 output_scaling=None,
                 constrained=False):
        
        super().__init__()
        # Create Neural Network
        layers = []
        layers.append(nn.Linear(in_dim, h_dim))
        for i in range(num_layers):
            if i < num_layers-1:
                layers.append(nn.Softplus())
            else:
                # last layer has tanh as activation function
                # which acts as a regulator
                layers.append(nn.Tanh())
                break
            layers.append(nn.Linear(h_dim, h_dim))
        layers.append(nn.Linear(h_dim, out_dim))
        self.layers = nn.Sequential(*layers)
        
        # Initialize controller with zeros in the last layer
        if zero_init: self._init_zeros()
        self.zero_init = zero_init
        
        # Scaling
        if constrained is False and output_scaling is not None:
            warn("Output scaling has no effect without the `constrained` variable set to true")
        if input_scaling is None:
            input_scaling = torch.ones(in_dim)
        if output_scaling is None:
            # scaling[:, 0] -> min value
            # scaling[:, 1] -> max value
            output_scaling = torch.cat([-torch.ones(out_dim)[:,None],
                                         torch.ones(out_dim)[:,None]], -1)
        self.in_scaling = input_scaling
        self.out_scaling = output_scaling
        self.constrained = constrained
        
    def forward(self, t, x):
        x = self.layers(self.in_scaling.to(x)*x)
        if self.constrained:
            # we consider the constraints between -1 and 1
            # and then we rescale them
            x = tanh(x)
            # x = torch.clamp(x, -1, 1) # not working in some applications # TODO: fix the tanh to clamp
            x = self._rescale(x)
        return x
    
    def _rescale(self, x):
        s = self.out_scaling.to(x)
        return 0.5*(x + 1)*(s[...,1]-s[...,0]) + s[...,0]
    
    def _reset(self):
        '''Reinitialize layers'''
        for p in self.layers.children():
            if hasattr(p, 'reset_parameters'):
                p.reset_parameters()
        if self.zero_init: self._init_zeros()

    def _init_zeros(self):
        '''Reinitialize last layer with zeros'''
        for p in self.layers[-1].parameters(): 
            nn.init.zeros_(p)
            

class RandConstController(nn.Module):
    """Constant controller
    We can use this for residual propagation and MPC steps (forward propagation)"""
    def __init__(self, shape=(1,1), u_min=-1, u_max=1):
        super().__init__()
        self.u0 = torch.Tensor(*shape).uniform_(u_min, u_max)
        
    def forward(self, t, x):
        return self.u0


class CartpoleIntegralCost(nn.Module):
    '''
    Integral cost function (simplified)
    We put most weight on the position and angular position
    '''
    def __init__(self, x_star):
        super().__init__()
        self.x_star = x_star
        
    def forward(self, x, u=torch.Tensor([0.])):
        cost = 6 * x[..., 0]**2 + 5*(x[..., 2])**2 + 0.1*x[..., 1]**2 + 0.1*x[..., 3]**2
        return cost.mean()
