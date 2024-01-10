import numpy as np
import torch
import math
from numpy import sin, cos, pi
import ipdb
from matplotlib import pyplot as plt 
from operator import itemgetter
from optimizers import newton_al
from utils import *

class Spaces:
    def __init__(self,shape, high=0, low=0):
        self.shape = shape
        self.high = torch.tensor(high)
        self.low = torch.tensor(low)
    def sample(self, ):
        return torch.rand((1,))*(self.high-self.low) + self.low



class PendulumDynamics(torch.nn.Module):
    def __init__(self,batch_size,device,action_clamp=True,l=1.0,m=1.0):
        super().__init__()
        self.max_speed = 8
        self.max_torque = 2.0
        self.dt = 0.05
        self.g = 10.0
        self.m = m
        self.l = l
        self.batch_size = batch_size
        self.device = device
        self.observation_space = Spaces((2,))
        self.action_space = Spaces((1,), (2.5,), (-2.5,))
        self.action_clamp=action_clamp
        self.env_name = 'pendulum_'
        self.action_coeffs = 0.05
        
    def forward(self, state, action, return_costs=False, split_costs=False, normalize=True):
        th = state[:, 0].view(-1, 1)
        thdot = state[:, 1].view(-1, 1)

        g = 10
        m = self.m
        l = self.l
        dt = 0.05

        u = action
        action_coeff = 0.1
        if self.action_clamp:
            u = torch.clamp(u, -2, 2)
            action_coeff = 0.001


        thetadoubledot = (-3 * g / (2 * l) * torch.sin(th + math.pi) + 3. / (m * l ** 2) * u)
        newthdot = thdot + thetadoubledot * dt
        newth = th + newthdot * dt
        newthdot = torch.clamp(newthdot, -8, 8)
        if normalize:
            state = torch.cat((angle_normalize(newth), newthdot), dim=1)
        else:
            state = torch.cat((newth, newthdot), dim=1)
        statedot = torch.cat((newthdot, thetadoubledot), dim=1)
        if return_costs:
            costs = 0.5*(angle_normalize(th) ** 2 + 0.1 * thdot ** 2 + action_coeff * (u ** 2))/20
            if split_costs:
                action_costs = 0.5 * action_coeff * (u**2)/20
                state_costs = 0.5*(angle_normalize(th) ** 2 + 0.1 * thdot ** 2)/20
                return (state, -costs.squeeze(dim=-1), -state_costs.squeeze(dim=1), -action_costs.squeeze(dim=1))
            return (state, -costs.squeeze(dim=-1), None)
        return state

    def forward_midpoint(self, state, action, return_costs=False):
        th = state[:, 0].view(-1, 1)
        thdot = state[:, 1].view(-1, 1)

        g = 10
        m = self.m
        l = self.l
        dt = 0.05

        u = action
        u = torch.clamp(u, -2, 2)

        thetadoubledot = (-3 * g / (2 * l) * torch.sin(th + math.pi) + 3. / (m * l ** 2) * u)
        newthdot = thdot + thetadoubledot * dt*0.5
        newth = th + newthdot * dt * 0.5
        thetadoubledot = (-3 * g / (2 * l) * torch.sin(newth + math.pi) + 3. / (m * l ** 2) * u)
        newthdot = thdot + thetadoubledot * dt
        newth = th + newthdot * dt
        # newthdot = torch.clamp(newthdot, -8, 8)

        state = torch.cat((angle_normalize(newth), newthdot), dim=1)
        statedot = torch.cat((newthdot, thetadoubledot), dim=1)
        if return_costs:
            costs = 0.5*(angle_normalize(th) ** 2 + 0.1 * thdot ** 2 + 0.001 * (u ** 2))/20
            return (state, -costs, _)
        return state

    def step(self, action, state=None, return_costs=True):
        # TODO: Need to give _get_obs() = [costheta, sintheta, thetadot] as output
        if state is None:
            state, rewards, _ = self.forward(self.state, action, return_costs=True)
        else:
            state, rewards, _ = self.forward(state, action, return_costs=True)
        self.state = state
        self.last_u = action  # for rendering
        return state, rewards, None, None
    
    def reset(self, idxs=None):
        # TODO: Need to give _get_obs() = [costheta, sintheta, thetadot] as output
        if idxs is None:
            # ipdb.set_trace()
            self.state = torch.rand((self.batch_size, 2)).to(self.device)*2 - 1
            self.state[:,0]*= np.pi#*0.0#01
            self.state[:,1]*= 8
            self.last_u = None
        else:
            # ipdb.set_trace()
            self.state[idxs] = torch.rand((sum(idxs), 2)).to(self.device)*2 - 1
            self.state[idxs,0]*=np.pi
        return self.state

    def get_frame(self, x, ax=None):
        l = self.l
        if len(x) == 2:
            th = x[0]
            cos_th = np.cos(th)
            sin_th = np.sin(th)
        elif len(x) == 3:
            cos_th, sin_th= x[0], x[1]
            th = np.arctan2(sin_th, cos_th)
        x = sin_th*l
        y = cos_th*l

        if ax is None:
            fig, ax = plt.subplots(figsize=(6,6))
        else:
            fig = ax.get_figure()

        ax.plot((0,x), (0, y), color='k')
        ax.set_xlim((-l*1.2, l*1.2))
        ax.set_ylim((-l*1.2, l*1.2))
        return fig, ax

    def sample_goal_state(self, radius=[0.001,0.001]):
        radius = torch.tensor(radius).unsqueeze(0).to(self.device)
        state = torch.rand((self.batch_size, 2)).to(self.device)*radius*2 - radius
        return state, torch.zeros_like(state[:, 0])+1e-8

    def sample_new_states(self, buffer, counts, curv, density, model):
        step_size = torch.tensor([1/5,8/5]).to(self.device)

        # counts_np = np.hstack(counts)
        counts_np = counts
        probs = np.exp(-counts_np)*np.exp(-density + density.min())#*np.exp(-1/val_errs)
        probs = probs/probs.sum()
        try:
            idxs = np.random.choice(a=len(buffer), size=self.batch_size*2, replace=False, p=probs)
        except:
            ipdb.set_trace()

        buffer_idxs = list(itemgetter(*idxs)(buffer))
        samples, _, _, _, _ = map(torch.stack, zip(*buffer_idxs))
        samples = torch.tensor(samples).to(self.device)
        samples.requires_grad_(True)
        V = model(samples)
        Vx = torch.autograd.grad(V.sum(), samples, retain_graph=True, create_graph=True)[0]
        dx = -Vx
        # ipdb.set_trace()
        dxHdx = (torch.autograd.grad(Vx, samples, grad_outputs=dx)[0]*dx).sum(dim=-1)
        chosen = dxHdx < dxHdx.mean()
        idxs_chosen = idxs[chosen.cpu().numpy()]
        # counts_idx_chosen = list(itemgetter(*idxs_chosen)(counts))
        counts[idxs_chosen] +=1
        dx_chosen = dx[chosen]/(dx[chosen].abs().max(dim=0)[0].unsqueeze(0))
        step_size_rand = torch.rand(dx_chosen.shape)*step_size.unsqueeze(0)
        # step_size_rand = step_size.unsqueeze(0)
        new_samples = samples[chosen] + dx_chosen*step_size_rand#.unsqueeze(0)
        # buffer = torch.cat([buffer, new_samples.detach().clone()], dim=0)
        # counts = torch.cat([counts, torch.zeros(new_samples.shape[0])], dim=0)
        # ipdb.set_trace()
        curv[idxs] = dxHdx.detach().clone().numpy()
        return new_samples.detach().clone(), dxHdx[chosen].detach().clone().numpy()

    def sample_new_states_ilqr_backward(self, buffer, counts, curv, density, model, args):
        step_size = torch.tensor([1/5,8/5]).to(self.device)

        # counts_np = np.hstack(counts)
        counts_np = counts
        probs = np.exp(-counts_np)*np.exp(-density + density.min())#*np.exp(-1/val_errs)
        probs = probs/probs.sum()
        try:
            idxs = np.random.choice(a=len(buffer), size=self.batch_size*2, replace=False, p=probs)
        except:
            ipdb.set_trace()

        buffer_idxs = list(itemgetter(*idxs)(buffer))
        samples, _, _, _, _ = map(torch.stack, zip(*buffer_idxs))
        rand_noise = torch.randn(samples.shape).to(self.device)*2 - 1
        rand_noise[:,0] *= np.pi/100
        rand_noise[:,1] *= 8/100
        samples = torch.tensor(samples).to(self.device) + rand_noise
        samples.requires_grad_(True)
        args.rand_dirs = None
        # num_steps = np.random.randint(3) + 1
        num_steps = 2
        new_samples = self.find_pre_samples(samples, args, nsteps=num_steps)
        new_samples[:, 0] = angle_normalize(new_samples[:, 0])
        new_samples[:, 1] = torch.clamp(new_samples[:, 1], -8, 8)
        # self.plot_pairs(samples.detach().cpu().numpy(), new_samples.detach().cpu().numpy())
        # ipdb.set_trace()
        return new_samples.detach().clone(), torch.zeros_like(new_samples[:, 0]).numpy()


    def sample_new_states_ilqr_backward_rrt(self, buffer, counts, curv, density, model, args, env, states):
        return sample_new_states_ilqr_backward_rrt(buffer, counts, curv, density, model, args, self.device, self.batch_size, env, states)

        # step_size = torch.tensor([1/5,8/5]).to(self.device)

        # # counts_np = np.hstack(counts)
        # # counts_np = counts
        # # probs = np.exp(-counts_np)*np.exp(-density + density.min())#*np.exp(-1/val_errs)
        # # probs = probs/probs.sum()
        # # try:
        # #     idxs = np.random.choice(a=len(buffer), size=self.batch_size*2, replace=False, p=probs)
        # # except:
        # #     ipdb.set_trace()

        # # buffer_idxs = list(itemgetter(*idxs)(buffer))
        # # samples, _, _, _, _ = map(torch.stack, zip(*buffer_idxs))


        # rand_states = torch.rand(self.batch_size, self.observation_space.shape[0]).to(self.device)*2 - 1
        # rand_states[:,0] *= np.pi
        # rand_states[:,1] *= 8
        # states = torch.stack(states)
        # rand_dist = (rand_states.unsqueeze(0) - states.unsqueeze(1)).norm(dim=-1)
        # min_dist_args = torch.argmin(rand_dist, dim=0)
        # samples = states[min_dist_args]
        # # ipdb.set_trace()
        # args.rand_dirs = rand_states - samples

        # rand_noise = torch.randn(samples.shape).to(self.device)*2 - 1
        # rand_noise[:,0] *= np.pi/100
        # rand_noise[:,1] *= 8/100

        # samples = torch.tensor(samples).to(self.device) + rand_noise
        # samples.requires_grad_(True)
        # # num_steps = np.random.randint(3) + 1
        # num_steps = 1

        # new_samples = self.find_pre_samples(samples, args, nsteps=num_steps)
        # new_samples[:, 0] = angle_normalize(new_samples[:, 0])
        # new_samples[:, 1] = torch.clamp(new_samples[:, 1], -8, 8)
        # # self.plot_pairs(samples.detach().cpu().numpy(), new_samples.detach().cpu().numpy())
        # # self.plot_triplets(samples.detach().cpu().numpy(), new_samples.detach().cpu().numpy(), rand_states)
        # # ipdb.set_trace()
        # return new_samples.detach().clone(), torch.zeros_like(new_samples[:, 0]).numpy()

    def plot_triplets(self, samples1, samples2, targs):
        # ipdb.set_trace()
        plt.scatter(samples1[:,0], samples1[:,1], c='r')
        plt.scatter(samples2[:,0], samples2[:,1], c='b')
        plt.scatter(targs[:,0], targs[:,1], c='g')
        for i in range(samples1.shape[0]):
            plt.plot([samples1[i,0], samples2[i,0], targs[i,0]], [samples1[i,1], samples2[i,1], targs[i,1]])
        # plt.quiver(samples1[:,0], samples1[:,1], samples2[:,0]-samples1[:,0], samples2[:,1]-samples1[:,1])
        plt.xlim([-0.3, 0.3])
        plt.ylim([-1, 1])
        plt.savefig('inv_ilqr_rrt_sample_triplets1r0.png')

    def plot_pairs(self, samples1, samples2):
        # ipdb.set_trace()
        plt.scatter(samples1[:,0], samples1[:,1], c='r')
        plt.scatter(samples2[:,0], samples2[:,1], c='b')
        for i in range(samples1.shape[0]):
            plt.plot([samples1[i,0], samples2[i,0]], [samples1[i,1], samples2[i,1]])
        # plt.quiver(samples1[:,0], samples1[:,1], samples2[:,0]-samples1[:,0], samples2[:,1]-samples1[:,1])
        # plt.xlim([-np.pi, np.pi])
        # plt.ylim([-8, 8])
        plt.savefig('inv_ilqr_rrt_sample_pairs1r0.png')

    def find_pre_samples(self, samples, args, nsteps=1):
        return find_pre_samples(samples, args, self, nsteps)
        # bsz = samples.shape[0]
        # new_samples = torch.stack([samples]*(nsteps+1), dim=1)
        # actions_init = torch.zeros((samples.shape[0], nsteps, 1))
        # args.rho = 1
        # args.rho_ratio = 10
        # args.mu_update_tol = 1e-3
        # args.rho_max = 1e6
        # if args.rand_dirs is None:
        #     rand_dirs = torch.rand(samples.shape)
        #     rand_dirs = rand_dirs/(rand_dirs.norm(dim=-1).unsqueeze(-1))
        # else:
        #     rand_dirs = args.rand_dirs
        #     rand_dirs = rand_dirs/(rand_dirs.norm(dim=-1).unsqueeze(-1))
        # prob = torch.rand(rand_dirs.shape[0]).to(rand_dirs)
        # self.mask = (prob < 1).to(prob)
        # result_info = newton_al(self.gradfn, self.lagr_func, actions_init.reshape(bsz, -1), new_samples.reshape(bsz, -1), samples, args, rand_dirs)
        # next_samples = result_info['result']
        # tzero_samples = next_samples[:, :self.observation_space.shape[0]]

        # return tzero_samples

    def gradfn(self, xu, mu, rho, sT, xsize, rand_dirs):
        # xu = torch.autograd.Variable(xu, requires_grad = True)
        return grad_func(xu, mu, rho, sT, xsize, rand_dirs, self.mask, self)

        # diff, lagr = self.lagr_func(xu, mu, rho, sT, xsize, rand_dirs)
        # lgrad = torch.autograd.grad(lagr, xu, retain_graph=True, create_graph=True)[0]
        # if torch.isnan(lgrad).sum():
        #     ipdb.set_trace()
        # return lgrad, diff, lagr

    def lagr_func(self, xu, mu, rho, sT, xsz, rand_dirs, vec=False):
        # We could maximize the dot product with the negative gradient direction w.r.t the value function?
        # Or we could minimize the cumulative sum of rewards but we would probably have to separate the 
        # state rewards and action rewards.
        return lagr_func(xu, mu, rho, sT, xsz, rand_dirs, self.mask, self, vec=False)

        # x, u = xu[:, :xsz], xu[:, xsz:]
        # bsz = x.shape[0]
        # x1 = x[:, :-self.observation_space.shape[0]].reshape(-1, self.observation_space.shape[0])
        # x2 = x[:, self.observation_space.shape[0]:].reshape(-1, self.observation_space.shape[0])
        # nx, r, rs, ra = self.forward(x1, u.reshape(-1,1), return_costs=True, split_costs=True)
        # dp = -torch.tanh(((x[:, :self.observation_space.shape[0]] - sT)*rand_dirs*10).sum(dim=-1))#/((x[:, :self.observation_space.shape[0]] - sT).norm(dim=-1)+1e-8).detach()
        # # dp = torch.nn.functional.cosine_similarity((x[:, :self.observation_space.shape[0]] - sT), rand_dirs, dim=-1)
        # # ipdb.set_trace()

        # rs_sum = rs.reshape(bsz, -1).sum(dim=1)
        # rs_sum = rs_sum*(1-self.mask) + self.mask*dp#*(rs.abs().mean()/(dp.abs().mean() + 1e-8)).detach()
        # if torch.isnan(rs).sum() > 0:
        #     ipdb.set_trace()
        # diff = torch.cat([(nx - x2).reshape((bsz, -1)), sT - x[:, -self.observation_space.shape[0]:]], dim=1)
        # if vec:
        #     # lagr = r.reshape(bsz, -1).sum(dim=1) + (mu * diff).sum(dim=1) + (rho * diff**2).sum(dim=1) + self.initial_state_cost(x[:, :self.observation_space.shape[0]], sT)
        #     lagr = rs_sum - ra.reshape(bsz, -1).sum(dim=1) + (mu * diff).sum(dim=1) + (rho * diff**2).sum(dim=1) + self.initial_state_cost(x[:, :self.observation_space.shape[0]], sT)
        # else:
        #     lagr = (rs_sum - ra.reshape(bsz, -1).sum(dim=1)).sum() + (mu * diff).sum() + (rho * diff**2).sum() + self.initial_state_cost(x[:, :self.observation_space.shape[0]], sT).sum()
        # # ipdb.set_trace()
        # return diff, lagr

    def initial_state_cost(self, x, sT):
        # need to add density cost and distance cost to the terminal state
        # We could find the k nearest neighbors of sT and use that to find the direction from sT
        # We could maximize the dot product with the negative gradient direction w.r.t the value function?
        # resolving ambuiguity - value gradient dot product
        return torch.zeros_like(x).sum(dim=1)


    def samples_from_buffer(self, buffer, states):
        bsz = states.shape[0]
        idxs = np.random.choice(a=len(buffer), size=bsz, replace=False)
        buffer_idxs = list(itemgetter(*idxs)(buffer))
        samples, _, _, _, _ = map(torch.stack, zip(*buffer_idxs))
        samples = torch.tensor(samples).to(states)
        return samples

    def plot_samples(self, buffer, model, epoch, samples, new_samples):
        plot_samples(buffer, model, epoch, self.batch_size, self.env_name, samples, new_samples)
        # sample a bunch of random points
        # 2d plot with value function at each of those points.
        # Compute ground truth value function using MPC cost to go and plot that using the same method. 
        
        # V = []
        # states = []
        # # model = model.cpu()
        # # ipdb.set_trace()
        # with torch.no_grad():
        # # if True:
        #     for i in range(len(buffer)//self.batch_size):
        #         state1 = [buffer_i[0] for buffer_i in buffer[i*self.batch_size: (i+1)*self.batch_size]]
        #         state = torch.stack(state1)
        #         # for buffer_i in buffer[i*self.batch_size: (i+1)*self.batch_size]:
        #             # for j in range(len(buffer_i)):
        #             # state1[j].append(buffer_i[j])
        #         # for j in range(len(state1)):
        #         #     try:
        #         #         torch.stack(state1[j])
        #         #     except:
        #         #         ipdb.set_trace()
        #         # try:
        #         #     state, _, _, _, _ = map(torch.stack, buffer[i*self.batch_size: (i+1)*self.batch_size])
        #         # except:
        #         #     ipdb.set_trace()
        #         states.append(state)
        #         V.append(model(state).squeeze().cpu())
        #         # if i%20==0:
        #         #     print(i, ) 
        # V = torch.cat(V, dim=0).cpu()*0 + 1
        # states = torch.cat(states, dim=0).cpu().detach().requires_grad_(False)
        # print("buffer size", states.shape[0])
        # # ipdb.set_trace()
        # plt.clf()
        # plt.scatter(states[:,0], states[:,1], c=V, s=2, cmap='gray')
        # plt.xlim([-np.pi, np.pi])
        # plt.ylim([-8, 8])
        # plt.savefig('pendulum_explore_backward/trial5' + '/val_states_{}.png'.format(epoch))

def angle_normalize(x, low=-math.pi, high=math.pi):
    return (((x - low) % (high-low)) + low)



class AcrobotEnv(torch.nn.Module):

    """
    Acrobot is a 2-link pendulum with only the second joint actuated.
    Initially, both links point downwards. The goal is to swing the
    end-effector at a height at least the length of one link above the base.
    Both links can swing freely and can pass by each other, i.e., they don't
    collide when they have the same angle.
    **STATE:**
    The state consists of the sin() and cos() of the two rotational joint
    angles and the joint angular velocities :
    [cos(theta1) sin(theta1) cos(theta2) sin(theta2) thetaDot1 thetaDot2].
    For the first link, an angle of 0 corresponds to the link pointing downwards.
    The angle of the second link is relative to the angle of the first link.
    An angle of 0 corresponds to having the same angle between the two links.
    A state of [1, 0, 1, 0, ..., ...] means that both links point downwards.
    **ACTIONS:**
    The action is either applying +1, 0 or -1 torque on the joint between
    the two pendulum links.
    .. note::
        The dynamics equations were missing some terms in the NIPS paper which
        are present in the book. R. Sutton confirmed in personal correspondence
        that the experimental results shown in the paper and the book were
        generated with the equations shown in the book.
        However, there is the option to run the domain with the paper equations
        by setting book_or_nips = 'nips'
    **REFERENCE:**
    .. seealso::
        R. Sutton: Generalization in Reinforcement Learning:
        Successful Examples Using Sparse Coarse Coding (NIPS 1996)
    .. seealso::
        R. Sutton and A. G. Barto:
        Reinforcement learning: An introduction.
        Cambridge: MIT press, 1998.
    .. warning::
        This version of the domain uses the Runge-Kutta method for integrating
        the system dynamics and is more realistic, but also considerably harder
        than the original version which employs Euler integration,
        see the AcrobotLegacy class.
    """

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 15}

    # dt = 0.05
    # dt = 0.01

    MAX_VEL_1 = 4 * pi
    MAX_VEL_2 = 9 * pi

    AVAIL_TORQUE = torch.tensor([-1.0, 0.0, +1])

    torque_noise_max = 0.0

    #: use dynamics equations from the nips paper or the book
    book_or_nips = "book"
    action_arrow = None
    domain_fig = None
    actions_num = 1
    # batch_size = 512

    num_obs = 4
    num_act = 1

    def __init__(self, batch_size=64, device=torch.device('cpu'), continuous=True, dt=0.01, T=400, l1=1.0, m1=1.0):
        super().__init__()
        self.viewer = None
        high = np.array(
            [1.0, 1.0, 1.0, 1.0, self.MAX_VEL_1, self.MAX_VEL_2], dtype=np.float32
        )
        low = -high
        self.continuous = continuous
        self.actions_disc = torch.arange(-6,7,3.0).unsqueeze(-1).to(device)
        # self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        # self.action_space = spaces.Discrete(3)
        self.state = None
        self.device = device
        self.batch_size = batch_size
        self.gs = torch.zeros((1,4)).to(device)
        self.gs[0,0] += np.pi/2
        self.init_state = torch.zeros((1,4)).to(device)
        self.init_state[0,0] -= np.pi/2

        self.observation_space = Spaces((4,))
        self.action_space = Spaces((1,), (25,), (-25,))
        self.env_name = 'acrobot_'
        self.action_coeffs = 0.05
        self.dt = dt
        self.in_state = self.reset()[:1]
        self.num_steps = 0
        self.T = T

        self.LINK_LENGTH_1 = l1  # [m]
        self.LINK_LENGTH_2 = 1.0  # [m]
        self.LINK_MASS_1 = m1  #: [kg] mass of link 1
        self.LINK_MASS_2 = 1.0  #: [kg] mass of link 2
        self.LINK_COM_POS_1 = 0.5  #: [m] position of the center of mass of link 1
        self.LINK_COM_POS_2 = 0.5  #: [m] position of the center of mass of link 2
        self.LINK_MOI = 1.0  #: moments of inertia for both links
        self.LINK_i1=.2
        self.LINK_i2=.8
        # self.seed()

    # def seed(self, seed=None):
        # self.np_random, seed = seeding.np_random(seed)
        # return [seed]

    # def reset(self):
        # self.state = self.np_random.uniform(low=-0.1, high=0.1, size=(4,)).astype(
        #     np.float32
        # )
        # return self._get_ob()
    
    # def forward1(self, s, a, return_sdot=False, un_state=False):
        # if not self.continuous:
        #     torque = self.actions_disc[a]
        # else:
        #     # torque = torch.round(a)
        #     torque = a
        #     # torque = torch.round(a*4)/4
        #     # torque = torch.clip(torque, -1, 1)


        # # # Add noise to the force action
        # # if self.torque_noise_max > 0:
        # #     torque += self.np_random.uniform(
        # #         -self.torque_noise_max, self.torque_noise_max
        # #     )

        # # Now, augment the state with our force action so it can be passed to
        # # _dsdt
        # # ipdb.set_trace()
        # s_augmented = torch.cat([s, torque], dim=1)

        # # ns = rk4(self._dsdt, s_augmented, [0, self.dt])
        # ns = euler(self._dsdt, s_augmented, [0, self.dt])

        # ns[:, :2] = angle_normalize(ns[:, :2])
        # # ns[:, 1] = angle_normalize(ns[:, 1])
        # ns[:, 2] = torch.clamp(ns[:, 2], -self.MAX_VEL_1, self.MAX_VEL_1)
        # ns[:, 3] = torch.clamp(ns[:, 3], -self.MAX_VEL_2, self.MAX_VEL_2)
        # terminal = self._terminal(ns)
        # # reward = -torch.ones_like(ns[:, :1])*(1-terminal)
        # coeff = [1,0.1] #[4,1] 
        # # ipdb.set_trace()
        # ns_dash = ns.clone()
        # ns_dash[:,0] = angle_denormalize(ns[:, 0])
        # gs_dash = self.gs.clone()
        # gs_dash[:,0] *=0
        # reward = (- coeff[0]*((ns_dash-gs_dash)*(ns-gs_dash)).sum(dim=-1) - coeff[1]*(a*a).sum(dim=-1))/100
        # return ns, reward, terminal

    def forward(self, s, a, return_costs=False, split_costs=False, normalize=True):

        if not self.continuous:
            torque = self.actions_disc[a]
        else:
            # torque = torch.round(a)*3
            torque = a
            # torque = a
            # torque = torch.round(a*4)/4
            # torque = torch.clip(torque, -6, 6)

        # ipdb.set_trace()
        # # Add noise to the force action
        # if self.torque_noise_max > 0:
        #     torque += self.np_random.uniform(
        #         -self.torque_noise_max, self.torque_noise_max
        #     )

        # Now, augment the state with our force action so it can be passed to
        # _dsdt
        # ipdb.set_trace()
        s_augmented = torch.cat([s, torque], dim=1)

        # ns = rk4(self._dsdt, s_augmented, [0, self.dt])
        # ns = rk4(self._dynamics, s_augmented, [0, self.dt])
        ns = euler(self._dynamics, s_augmented, [0, self.dt])
        # ipdb.set_trace()
        if normalize:
            ns_th0 = angle_normalize(ns[:, 0], 0, 2*math.pi)
            ns_th1 = angle_normalize(ns[:, 1], -math.pi, math.pi)
        else:
            ns_th0 = ns[:, 0]
            ns_th1 = ns[:, 1]
        ns_vel2 = torch.clamp(ns[:, 2], -self.MAX_VEL_1, self.MAX_VEL_1)
        ns_vel3 = torch.clamp(ns[:, 3], -self.MAX_VEL_2, self.MAX_VEL_2)
        ns = torch.stack([ns_th0, ns_th1, ns_vel2, ns_vel3], dim=1)
        terminal = self._terminal(ns)
        # reward = -torch.ones_like(ns[:, :1])*(1-terminal)
        coeff = [1,0.01] #[4,1] 
        # ipdb.set_trace()

        # ns_dash = ns.clone()
        # ns_dash[:,0] = angle_denormalize(ns[:, 0])
        # gs_dash = self.gs.clone()
        # gs_dash[:,0] *=0
        # reward = (- coeff[0]*((ns_dash-gs_dash)*(ns-gs_dash)).sum(dim=-1))# - coeff[1]*(a*a).sum(dim=-1))#/100
        if return_costs:
            reward = torch.sin(ns[:, 0]) + torch.sin(ns[:, 0] + ns[:, 1])
            if split_costs:
                state_reward = reward
                action_reward = reward*0
                return (ns, reward, state_reward, action_reward)
            return ns, reward, terminal
        return ns

    def step(self, action, state=None, return_costs=True):
        if state is None:
            ns, reward, terminal = self.forward(self.state, action, return_costs=True)
        else:
            ns, reward, terminal = self.forward(state, action, return_costs=True)
        self.state = ns
        return (ns, reward, terminal, {})

    def reset(self, idxs=None):
        # TODO: Need to give _get_obs() = [costheta, sintheta, thetadot] as output
        if idxs is None:
            # ipdb.set_trace()
            self.state = torch.rand((self.batch_size, 4)).to(self.device)*2 - 1
            self.state[:,0]*= np.pi
            self.state[:,0]+= np.pi
            self.state[:,1]*= np.pi
            self.state[:,2]*= self.MAX_VEL_1
            self.state[:,3]*= self.MAX_VEL_2
            self.last_u = None
        else:
            # ipdb.set_trace()
            self.state[idxs] = torch.rand((sum(idxs), 4)).to(self.device)*2 - 1
            self.state[idxs,0]*= np.pi
            self.state[idxs,0]+= np.pi
            self.state[idxs,1]*= np.pi
            self.state[idxs,2]*= self.MAX_VEL_1
            self.state[idxs,3]*= self.MAX_VEL_2
        return self.state

    def reset_init(self, idxs=None):
        # TODO: Need to give _get_obs() = [costheta, sintheta, thetadot] as output
        self.state = torch.cat([self.init_state]*self.batch_size, dim=0)
        return self.state

    def _get_ob(self):
        s = self.state
        return np.array(
            [cos(s[0]), sin(s[0]), cos(s[1]), sin(s[1]), s[2], s[3]], dtype=np.float32
        )

    def _terminal(self, s):
        # s = self.state
        return (-torch.cos(s[:, 0]) - torch.cos(s[:, 1] + s[:, 0]) > 1.0).float().unsqueeze(-1)

    def _dsdt(self, s_augmented):
        m1 = self.LINK_MASS_1
        m2 = self.LINK_MASS_2
        l1 = self.LINK_LENGTH_1
        lc1 = self.LINK_COM_POS_1
        lc2 = self.LINK_COM_POS_2
        I1 = self.LINK_i1 #MOI
        I2 = self.LINK_I2 #MOI
        g = 9.8
        a = s_augmented[:, -1]
        s = s_augmented[:, :-1]
        theta1 = s[:, 0]
        theta2 = s[:, 1]
        dtheta1 = s[:, 2]
        dtheta2 = s[:, 3]
        d1 = (
            m1 * lc1 ** 2
            + m2 * (l1 ** 2 + lc2 ** 2 + 2 * l1 * lc2 * torch.cos(theta2))
            + I1
            + I2
        )
        d2 = m2 * (lc2 ** 2 + l1 * lc2 * torch.cos(theta2)) + I2
        phi2 = m2 * lc2 * g * torch.cos(theta1 + theta2 - pi / 2.0)
        phi1 = (
            -m2 * l1 * lc2 * dtheta2 ** 2 * torch.sin(theta2)
            - 2 * m2 * l1 * lc2 * dtheta2 * dtheta1 * torch.sin(theta2)
            + (m1 * lc1 + m2 * l1) * g * torch.cos(theta1 - pi / 2)
            + phi2
        )
        if self.book_or_nips == "nips":
            # the following line is consistent with the description in the
            # paper
            ddtheta2 = (a + d2 / d1 * phi1 - phi2) / (m2 * lc2 ** 2 + I2 - d2 ** 2 / d1)
        else:
            # the following line is consistent with the java implementation and the
            # book
            ddtheta2 = (
                a + d2 / d1 * phi1 - m2 * l1 * lc2 * dtheta1 ** 2 * torch.sin(theta2) - phi2
            ) / (m2 * lc2 ** 2 + I2 - d2 ** 2 / d1)
        ddtheta1 = -(d2 * ddtheta2 + phi1) / d1
        return torch.stack([dtheta1, dtheta2, ddtheta1, ddtheta2, ddtheta2*0.], dim=1)


    def _dynamics(self, s_augmented):
        m1 = self.LINK_MASS_1
        m2 = self.LINK_MASS_2
        l1 = self.LINK_LENGTH_1
        lc1 = self.LINK_COM_POS_1
        lc2 = self.LINK_COM_POS_2
        i1 = self.LINK_i1
        i2 = self.LINK_i2
        g = 9.8
        s0 = s_augmented[:, :-1]
        act = s_augmented[:, -1]
        # print("1:s0:", s0)
        tau = act#.item()
        th1 = s0[:, 0]
        th2 = s0[:, 1]
        th1d = s0[:, 2]
        th2d = s0[:, 3]
        g = 9.8
        # ipdb.set_trace()
        TAU = torch.stack([torch.zeros_like(tau),tau], dim=1).unsqueeze(-1)

        m11 = m1*lc1**2 + m2*(l1**2 + lc2**2 + 2*l1*lc2*torch.cos(th2)) + i1 + i2
        m22 = m2*lc2**2 + i2
        m12 = m2*(lc2**2 + l1*lc2*torch.cos(th2)) + i2
        M = torch.stack([torch.stack([m11, m12], dim=-1), torch.stack([m12, m22*torch.ones_like(m12)], dim=-1)], dim=-2)
        # print("1:M:", M)
        h1 = -m2*l1*lc2*torch.sin(th2)*th2d**2 - 2*m2*l1*lc2*torch.sin(th2)*th2d*th1d
        h2 = m2*l1*lc2*torch.sin(th2)*th1d**2
        H = torch.stack([h1,h2],dim=-1).unsqueeze(-1)
        # print("1:H:", H)

        phi1 = (m1*lc1+m2*l1)*g*torch.cos(th1) + m2*lc2*g*torch.cos(th1+th2)
        phi2 = m2*lc2*g*torch.cos(th1+th2)
        PHI = torch.stack([phi1, phi2], dim=-1).unsqueeze(-1)
        # print("1:PHI:", PHI)

        d2th = torch.linalg.solve(M,(TAU - H - PHI)).squeeze(-1)
        # print("1:d2th:", d2th)
        return torch.stack([th1d, th2d, d2th[:,0], d2th[:,1], th1d*0], dim=1)

    def sample_goal_state(self, radius=[0.001,0.001,0.001,0.001]):
        radius = torch.tensor(radius).unsqueeze(0).to(self.device)
        state = torch.rand((self.batch_size, 4)).to(self.device)*radius*2 - radius
        state[:,0] += math.pi/2
        return state, torch.zeros_like(state[:, 0])+1e-8

    def get_frame(self, s, ax=None):
        # l = self.l
        # if len(x) == 2:
        #     th = x[0]
        #     cos_th = np.cos(th)
        #     sin_th = np.sin(th)
        # elif len(x) == 3:
        #     cos_th, sin_th= x[0], x[1]
        #     th = np.arctan2(sin_th, cos_th)
        p1 = [-self.LINK_LENGTH_1 * cos(s[0]), self.LINK_LENGTH_1 * sin(s[0])]

        p2 = [
            p1[0] - self.LINK_LENGTH_2 * cos(s[0] + s[1]),
            p1[1] + self.LINK_LENGTH_2 * sin(s[0] + s[1]),
        ]
        bound = self.LINK_LENGTH_1 + self.LINK_LENGTH_2 + 0.2

        if ax is None:
            fig, ax = plt.subplots(figsize=(2*bound,2*bound))
        else:
            fig = ax.get_figure()

        ax.plot((0,p1[0]), (0, p1[1]), color='k')
        ax.plot((p1[0],p2[0]), (p1[1], p2[1]), color='k')
        ax.set_xlim((-bound*1.1, bound*1.1))
        ax.set_ylim((-bound*1.1, bound*1.1))
        return fig, ax

    def render(self, mode="human"):
        from gym.envs.classic_control import rendering

        s = self.state

        if self.viewer is None:
            self.viewer = rendering.Viewer(500, 500)
            bound = self.LINK_LENGTH_1 + self.LINK_LENGTH_2 + 0.2  # 2.2 for default
            self.viewer.set_bounds(-bound, bound, -bound, bound)

        if s is None:
            return None

        p1 = [-self.LINK_LENGTH_1 * cos(s[0]), self.LINK_LENGTH_1 * sin(s[0])]

        p2 = [
            p1[0] - self.LINK_LENGTH_2 * cos(s[0] + s[1]),
            p1[1] + self.LINK_LENGTH_2 * sin(s[0] + s[1]),
        ]

        xys = np.array([[0, 0], p1, p2])[:, ::-1]
        thetas = [s[0] - pi / 2, s[0] + s[1] - pi / 2]
        link_lengths = [self.LINK_LENGTH_1, self.LINK_LENGTH_2]

        self.viewer.draw_line((-2.2, 1), (2.2, 1))
        for ((x, y), th, llen) in zip(xys, thetas, link_lengths):
            l, r, t, b = 0, llen, 0.1, -0.1
            jtransform = rendering.Transform(rotation=th, translation=(x, y))
            link = self.viewer.draw_polygon([(l, b), (l, t), (r, t), (r, b)])
            link.add_attr(jtransform)
            link.set_color(0, 0.8, 0.8)
            circ = self.viewer.draw_circle(0.1)
            circ.set_color(0.8, 0.8, 0)
            circ.add_attr(jtransform)

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def sample_new_states_ilqr_backward(self, buffer, counts, curv, density, model, args, env):
        return sample_new_states_ilqr_backward(buffer, counts, curv, density, model, args, self.device, self.batch_size, env)

    def sample_new_states_ilqr_backward_rrt(self, buffer, counts, curv, density, model, args, env, states, policy):
        num_on_policy = 0
        if len(buffer) > 5200:
            num_on_policy = self.batch_size
        num_backward = self.batch_size - num_on_policy

        if num_on_policy >0:
            new_samples_play, samples_play = self.play(num_on_policy, policy)
            new_samples = new_samples_play
            samples = samples_play
        
        if num_backward > 0:
            new_samples, samples, curv = sample_new_states_ilqr_backward_rrt(buffer, counts, curv, density, model, args, self.device, num_backward, env, states)
            # new_samples = torch.cat([new_samples, new_samples_play], dim=0)
            # samples = torch.cat([samples, samples_play], dim=0)
        return new_samples, samples, torch.zeros_like(new_samples[:, 0]).numpy()

    def play(self, num_on_policy, policy):
        samples = [self.in_state]
        for i in range(num_on_policy):
            action = policy(self.in_state)
            next_state = self.forward(self.in_state, action) # Step
            samples.append(next_state)
            self.in_state = next_state
            self.num_steps +=1
            if self.num_steps >= self.T:
                self.num_steps = 0
                self.in_state = self.reset()[:1]
        new_samples = torch.cat(samples[1:], dim=0)
        samples = torch.cat(samples[:-1], dim=0)
        return new_samples, samples 
    
    def plot_samples(self, buffer, model, epoch, samples, new_samples):
        plot_samples(buffer, model, epoch, self.batch_size, self.env_name, samples, new_samples)

def angle_denormalize(x):
    mask = (x>0).float()
    x = (math.pi - x)*mask + (-math.pi-x)*(1-mask)
    return x

def rk4(derivs, y0, t):
    """
    Integrate 1D or ND system of ODEs using 4-th order Runge-Kutta.
    This is a toy implementation which may be useful if you find
    yourself stranded on a system w/o scipy.  Otherwise use
    :func:`scipy.integrate`.

    Args:
        derivs: the derivative of the system and has the signature ``dy = derivs(yi)``
        y0: initial state vector
        t: sample times
        args: additional arguments passed to the derivative function
        kwargs: additional keyword arguments passed to the derivative function

    Example 1 ::
        ## 2D system
        def derivs(x):
            d1 =  x[0] + 2*x[1]
            d2 =  -3*x[0] + 4*x[1]
            return (d1, d2)
        dt = 0.0005
        t = arange(0.0, 2.0, dt)
        y0 = (1,2)
        yout = rk4(derivs6, y0, t)

    If you have access to scipy, you should probably be using the
    scipy.integrate tools rather than this function.
    This would then require re-adding the time variable to the signature of derivs.

    Returns:
        yout: Runge-Kutta approximation of the ODE
    """

    # try:
    #     Ny = len(y0)
    # except TypeError:
    #     yout = np.zeros((len(t),), np.float_)
    # else:
    #     yout = np.zeros((len(t), Ny), np.float_)

    # yout = torch.zeros((len(t),y0.shape[0], y0.shape[1])).to(y0)
    yout = []
    yout.append(y0)

    for i in np.arange(len(t) - 1):

        thist = t[i]
        dt = t[i + 1] - thist
        dt2 = dt / 2.0
        y0 = yout[i]
        # y0.requires_grad_(True)
        k1 = derivs(y0)
        # ipdb.set_trace()
        k2 = derivs(y0 + dt2 * k1)
        k3 = derivs(y0 + dt2 * k2)
        k4 = derivs(y0 + dt * k3)
        yout.append(y0 + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4))
    # We only care about the final timestep and we cleave off action value which will be zero
    # ipdb.set_trace()
    return yout[-1][:, :4]

def euler(derivs, y0, t):

    yout = []
    yout.append(y0)

    for i in np.arange(len(t) - 1):

        thist = t[i]
        dt = t[i + 1] - thist
        y0 = yout[i]
        k1 = derivs(y0)
        yout.append(y0 + dt * k1)
    # We only care about the final timestep and we cleave off action value which will be zero
    # ipdb.set_trace()
    return yout[-1][:, :4]


class SGAcroEnvWrapper(torch.nn.Module):
    def __init__(self, batch_size=64, device=torch.device('cpu')):
        super().__init__()
        self.env = SGAcroEnv()
        self.batch_size = batch_size
        self.device = device
        self.observation_space = Spaces((4,))
        self.action_space = Spaces((1,), (25,), (-25,))
        self.MAX_VEL_1 = 4 * pi
        self.MAX_VEL_2 = 9 * pi

    def forward(self, s, a, return_costs=False):
        next_states = []
        rewards = []
        for i in range(s.shape[0]):
            ns, r, terminal, _ = self.env.forward1(s[i].detach().cpu().numpy(),a[i].detach().cpu().numpy())
            next_states.append(torch.tensor(ns).to(s))
            rewards.append(torch.tensor(r).to(s))
        ns = torch.stack(next_states, dim=0)
        reward = torch.stack(rewards, dim=0)
        if return_costs:
            return ns, reward, terminal
        return ns

    def step(self, action, state=None, return_costs=True):
        if state is None:
            ns, reward, terminal = self.forward(self.state, action, return_costs=True)
        else:
            ns, reward, terminal = self.forward(state, action, return_costs=True)
        self.state = ns
        return (ns, reward, terminal, {})

    def reset(self, idxs=None):
        # TODO: Need to give _get_obs() = [costheta, sintheta, thetadot] as output
        if idxs is None:
            # ipdb.set_trace()
            self.state = torch.rand((self.batch_size, 4)).to(self.device)*2 - 1
            self.state[:,0]*= np.pi
            self.state[:,1]*= np.pi
            self.state[:,2]*= self.MAX_VEL_1
            self.state[:,3]*= self.MAX_VEL_2
            self.last_u = None
        else:
            # ipdb.set_trace()
            self.state[idxs] = torch.rand((sum(idxs), 4)).to(self.device)*2 - 1
            self.state[idxs,0]*= np.pi
            self.state[idxs,1]*= np.pi
            self.state[idxs,2]*= self.MAX_VEL_1
            self.state[idxs,3]*= self.MAX_VEL_2
        return self.state



def wrap(x, low, high):
    """
    :param x: a scalar
    :param m: minimum possible value in range
    :param M: maximum possible value in range
    Wraps ``x`` so m <= x <= M; but unlike ``bound()`` which
    truncates, ``wrap()`` wraps x around the coordinate system defined by m,M.\n
    For example, m = -180, M = 180 (degrees), x = 360 --> returns 0.
    """

    #    return x
    # x = (x + np.pi) % (2 * np.pi) - np.pi
    return (x - low) % (high - low) + low

def rk4_np(derivs, a, t0, dt, s0):
    """
    Single step of an RK4 solver, designed for control applications, so it passed an action to your
    derivs fcn
    Attributes:
        derivs: the function you are trying to integrate, should have signature:
        function(t,s,a) -> ds/dt
        a: action, should belong to the action space of your environment
        t0: float, initial time, often you can just set this to zero if all that matters for your
        derivs is the state and dt
        dt: how big of a timestep to integrate
        s0: initial state of your system, should belong to the envs obvervation space
    Returns:
        s[n+1]: I.E. the state of your system after integrating with action a for dt seconds
    Example:
        derivs = lambda t,q,a: (q+a)**2
        a =  1.0
        t0 = 0
        dt = .1
        s0 = 5
        s1 = rk4(derivs, a, t0, dt, s0)
    """
    k1 = dt * derivs(t0, s0, a)
    k2 = dt * derivs(t0 + dt / 2, s0 + k1 / 2, a)
    k3 = dt * derivs(t0 + dt / 2, s0 + k2 / 2, a)
    k4 = dt * derivs(t0 + dt, s0 + k3, a)

    return s0 + 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)




class SGAcroEnv(object):
    """ A simple acrobot environment
    """

    def __init__(self,
                 max_torque=25,
                 init_state=np.array([-pi/2, 0.0, 0.0, 0.0]),
                 init_state_weights=np.array([0.0, 0.0, 0.0, 0.0]),
                 dt=.05,
                 max_t=5,
                 act_hold=1,
                 integrator=rk4_np,
                 reward_fn=lambda ns, a: ((np.sin(ns[0]) + np.sin(ns[0] + ns[1])), False),
                 th1_range=[0, 2 * pi],
                 th2_range=[-pi, pi],
                 max_th1dot=float('inf'),
                 max_th2dot=float('inf'),
                 m1=1,
                 m2=1,
                 l1=1,
                 lc1=.5,
                 lc2=.5,
                 i1=.2,
                 i2=.8
                 ):
        """
        Args:
            max_torque: torque at which the controller saturates (N*m)
            init_state: initial state for the Acrobot. All zeros has the acrobot in it's stable equilibrium
            init_state_weights: initial state is going to be init_state + np.random.random(4)*init_state_weights
            dt: dt used by the simulator
            act_hold: like a frame skip, how many dts to hold every action for
            reward_fn: lambda that defines the reward function as a function of only the state variables
        """

        self.init_state = np.asarray(init_state, dtype=np.float64)
        self.init_state_weights = np.asarray(init_state_weights, dtype=np.float64)
        self.dt = dt
        self.max_t = max_t
        self.act_hold = act_hold
        self.int_accuracy = .01
        self.reward_fn = reward_fn
        self.integrator = integrator

        self.max_th1dot = max_th1dot
        self.max_th2dot = max_th2dot
        self.th1_range = th1_range
        self.th2_range = th2_range
        self.max_torque = max_torque

        self.m1 = m1
        self.m2 = m2
        self.l1 = l1
        self.lc1 = lc1
        self.lc2 = lc2
        self.i1 = i1
        self.i2 = i2

        # These are only used for rendering
        self.render_length1 = .5
        self.render_length2 = 1.0
        self.viewer = None

        low = np.array([th1_range[0], th2_range[0], -max_th1dot, -max_th2dot], dtype=np.float32)
        high = np.array([th1_range[1], th2_range[1], max_th1dot, max_th2dot], dtype=np.float32)

        # self.observation_space = spaces.Box(low=low-1, high=high+1, dtype=np.float32)
        # self.action_space = spaces.Box(low=np.array([-max_torque],dtype=np.float32), high=np.array([max_torque], dtype=np.float32), dtype=np.float32)

        self.t = 0

        self.num_steps = int(self.max_t / (self.act_hold * self.dt))
        # self.state = self.reset()

    def reset(self, init_vec=None):
        init_state = self.init_state
        init_state += np.random.random(4)*(self.init_state_weights * 2) - self.init_state_weights

        if init_vec is not None:
            init_state[0] = init_vec[0]
            init_state[1] = init_vec[1]
            init_state[2] = init_vec[2]
            init_state[3] = init_vec[3]

        self.t = 0
        self.state = init_state
        return self._get_obs()

    def reset_goal(self, init_vec=None):
        init_state = self.init_state
        init_state += np.random.random(4)*(self.init_state_weights * 2) - self.init_state_weights

        if init_vec is not None:
            init_state[0] = init_vec[0]
            init_state[1] = init_vec[1]
            init_state[2] = init_vec[2]
            init_state[3] = init_vec[3]

        self.t = 0
        self.state = init_state
        return self._get_obs()

    def forward1(self, s, a):
        a = np.clip(a, -self.max_torque, self.max_torque)
        self.t += self.dt * self.act_hold
        full_state = []

        for _ in range(self.act_hold):
            s = self.integrator(self._dynamics, a, self.t, self.dt, s)
            full_state.append(s)
        done = False
        reward, done = self.reward_fn(s, a)

        # I should probably do this with a wrapper...

        # if self.t >= self.max_t:
        #     done = True

        # if abs(s[2]) > self.max_th1dot or abs(s[2] > self.max_th2dot):
        #     reward -= 5
        #     done = True

        full_state = np.array(full_state).reshape(-1, 4)
        return self.get_obs(s), reward, done, {"full_state": full_state}

    def step(self, a):
        a = np.clip(a, -self.max_torque, self.max_torque)
        self.t += self.dt * self.act_hold
        full_state = []

        for _ in range(self.act_hold):
            self.state = self.integrator(self._dynamics, a, self.t, self.dt, self.state)
            full_state.append(self.state)

        done = False
        reward, done = self.reward_fn(self.state, a)

        # I should probably do this with a wrapper...
        if self.t >= self.max_t:
            done = True

        if abs(self.state[2]) > self.max_th1dot or abs(self.state[2] > self.max_th2dot):
            reward -= 5
            done = True

        full_state = np.array(full_state).reshape(-1, 4)
        return self._get_obs(), reward, done, {"full_state": full_state}

    def _get_obs(self):
        obs = self.state.copy()
        obs[0] = wrap(obs[0], self.th1_range[0], self.th1_range[1])
        obs[1] = wrap(obs[1], self.th2_range[0], self.th2_range[1])
        return obs

    def get_obs(self, state):
        obs = state
        obs[0] = wrap(obs[0], self.th1_range[0], self.th1_range[1])
        obs[1] = wrap(obs[1], self.th2_range[0], self.th2_range[1])
        return obs

    # render is shamelessly pulled from openAI's Acrobot-v1 env
    def render(self, mode='human'):
        from gym.envs.classic_control import rendering

        s = self.state

        if self.viewer is None:
            self.viewer = rendering.Viewer(500,500)
            bound = self.render_length1 + self.render_length2 + 0.2  # 2.2 for default
            self.viewer.set_bounds(-bound,bound,-bound,bound)

        if s is None: return None

        p1 = [self.render_length1 *
              sin(s[0]), self.render_length1 * cos(s[0])]

        p2 = [p1[0] + self.render_length2 * sin(s[0] + s[1]),
              p1[1] + self.render_length2 * cos(s[0] + s[1])]

        xys = np.array([[0,0], p1, p2])[:,::-1]
        thetas = [s[0], s[0]+s[1]]
        link_lengths = [self.render_length1, self.render_length2]

        self.viewer.draw_line((-2.2, 1), (2.2, 1))
        for ((x,y),th,llen) in zip(xys, thetas, link_lengths):
            l,r,t,b = 0, llen, .1, -.1
            jtransform = rendering.Transform(rotation=th, translation=(x,y))
            link = self.viewer.draw_polygon([(l,b), (l,t), (r,t), (r,b)])
            link.add_attr(jtransform)
            link.set_color(0,.8, .8)
            circ = self.viewer.draw_circle(.1)
            circ.set_color(.8, .8, 0)
            circ.add_attr(jtransform)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def _dynamics(self, t, s0, act):

        tau = act.item()
        th1 = s0[0]
        th2 = s0[1]
        th1d = s0[2]
        th2d = s0[3]
        g = 9.8
        # print("2:s0:", s0)
        # ipdb.set_trace()
        TAU = np.array([[0],[tau]])

        m11 = self.m1*self.lc1**2 + self.m2*(self.l1**2 + self.lc2**2 + 2*self.l1*self.lc2*cos(th2)) + self.i1 + self.i2
        m22 = self.m2*self.lc2**2 + self.i2
        m12 = self.m2*(self.lc2**2 + self.l1*self.lc2*cos(th2)) + self.i2
        M = np.array([[m11, m12], [m12, m22]])
        # print("2:M:", M)

        h1 = -self.m2*self.l1*self.lc2*sin(th2)*th2d**2 - 2*self.m2*self.l1*self.lc2*sin(th2)*th2d*th1d
        h2 = self.m2*self.l1*self.lc2*sin(th2)*th1d**2
        H = np.array([[h1],[h2]])
        # print("2:H:", H)

        phi1 = (self.m1*self.lc1+self.m2*self.l1)*g*cos(th1) + self.m2*self.lc2*g*cos(th1+th2)
        phi2 = self.m2*self.lc2*g*cos(th1+th2)
        PHI = np.array([[phi1], [phi2]])
        # print("2:PHI:", PHI)

        d2th = np.linalg.solve(M,(TAU - H - PHI))
        # print("2:d2th:", d2th)
        return np.array([th1d, th2d, d2th[0].item(), d2th[1].item()])




