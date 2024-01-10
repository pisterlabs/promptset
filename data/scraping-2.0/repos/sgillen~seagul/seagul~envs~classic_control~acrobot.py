"""classic Acrobot task"""
from gym import core, spaces
from gym.utils import seeding

import numpy as np
from numpy import pi
from numpy import sin, cos, pi

from seagul.integration import rk4, euler,wrap


class SGAcroEnv(core.Env):
    """ A simple acrobot environment
    """

    def __init__(self,
                 max_torque=25,
                 init_state=np.array([-pi/2, 0.0, 0.0, 0.0]),
                 init_state_weights=np.array([0.0, 0.0, 0.0, 0.0]),
                 dt=.01,
                 max_t=5,
                 act_hold=1,
                 integrator=euler,
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

        self.observation_space = spaces.Box(low=low-1, high=high+1, dtype=np.float32)
        self.action_space = spaces.Box(low=np.array([-max_torque],dtype=np.float32), high=np.array([max_torque], dtype=np.float32), dtype=np.float32)

        self.t = 0

        self.num_steps = int(self.max_t / (self.act_hold * self.dt))
        self.state = self.reset()

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

    # Shamelessly pulled from openAI's Acrobot-v1 env
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

        TAU = np.array([[0],[tau]])

        m11 = self.m1*self.lc1**2 + self.m2*(self.l1**2 + self.lc2**2 + 2*self.l1*self.lc2*cos(th2)) + self.i1 + self.i2
        m22 = self.m2*self.lc2**2 + self.i2
        m12 = self.m2*(self.lc2**2 + self.l1*self.lc2*cos(th2)) + self.i2
        M = np.array([[m11, m12], [m12, m22]])

        h1 = -self.m2*self.l1*self.lc2*sin(th2)*th2d**2 - 2*self.m2*self.l1*self.lc2*sin(th2)*th2d*th1d
        h2 = self.m2*self.l1*self.lc2*sin(th2)*th1d**2
        H = np.array([[h1],[h2]])

        phi1 = (self.m1*self.lc1+self.m2*self.l1)*g*cos(th1) + self.m2*self.lc2*g*cos(th1+th2)
        phi2 = self.m2*self.lc2*g*cos(th1+th2)
        PHI = np.array([[phi1], [phi2]])

        d2th = np.linalg.solve(M,(TAU - H - PHI))
        return np.array([th1d, th2d, d2th[0].item(), d2th[1].item()])

    # def _dynamics(self, t, s0, act):
    #     m1 = self.m1
    #     m2 = self.m2
    #     l1 = self.l1
    #     lc1 = self.lc1
    #     lc2 = self.lc2
    #     I1 = self.i1
    #     I2 = self.i2
    #     g = 9.8
    #     s = s0
    #     a = act.item()
    #     theta1 = s[0]+pi/2
    #     theta2 = s[1]
    #     dtheta1 = s[2]
    #     dtheta2 = s[3]
    #     d1 = m1 * lc1 ** 2 + m2 * (l1 ** 2 + lc2 ** 2 + 2 * l1 * lc2 * np.cos(theta2)) + I1 + I2
    #     d2 = m2 * (lc2 ** 2 + l1 * lc2 * np.cos(theta2)) + I2
    #     phi2 = m2 * lc2 * g * np.cos(theta1 + theta2 - np.pi / 2.0)
    #     phi1 = (
    #             -m2 * l1 * lc2 * dtheta2 ** 2 * np.sin(theta2)
    #             - 2 * m2 * l1 * lc2 * dtheta2 * dtheta1 * np.sin(theta2)
    #             + (m1 * lc1 + m2 * l1) * g * np.cos(theta1 - np.pi / 2)
    #             + phi2
    #     )
    #     # the following line is consistent with the description in the
    #     # paper
    #     #ddtheta2 = (a + d2 / d1 * phi1 - phi2) / (m2 * lc2 ** 2 + I2 - d2 ** 2 / d1)
    #     # else:
    #     #     # the following line is consistent with the java implementation and the
    #     #     # book
    #     ddtheta2 = (a + d2 / d1 * phi1 - m2 * l1 * lc2 * dtheta1 ** 2 * np.sin(theta2) - phi2) / (
    #         m2 * lc2 ** 2 + I2 - d2 ** 2 / d1
    #     )
    #     ddtheta1 = -(d2 * ddtheta2 + phi1) / d1
    
    #     return np.array([dtheta1, dtheta2, ddtheta1, ddtheta2])
