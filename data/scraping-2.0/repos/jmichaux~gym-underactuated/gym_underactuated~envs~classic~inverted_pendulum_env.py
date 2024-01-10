"""
Modified from OpenAI to match the cartpole system implemented by
Russ Tedrake in his awesome book Underactuated Robotics.  Unlike
Russ' book, here we are using moment of inertia
"""

import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import autograd.numpy as anp
from autograd import jacobian
from os import path

class InvertedPendulumEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self, masscart=1.0, masspole=0.1, total_length=1.0, tau=0.02, task="swingup"):
        # set task
        self.task = task
        self.g = self.gravity = 9.8
        self.masscart = masscart
        self.masspole = masspole
        self.total_mass = (self.masspole + self.masscart)
        self.L = total_length
        self.l = self.length =self.L / 2
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 10.0
        self.tau = tau
        self.I = (1/12) * self.masspole * (2*self.length)**2
        self.I = 0
        self.inertial = self.I + self.masspole * self.length**2
        self.b = 0.0
        self.n_coords = 2

        # precompute the jacobian of the dynamics
        self.jacobian = self._jacobian()

        # Angle at which to fail the episode
        if self.task == "balance":
            self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.theta_threshold_radians = 12 * 2 * math.pi / 360    
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
        high = np.array([
            self.x_threshold * 2,
            np.finfo(np.float32).max,
            self.theta_threshold_radians * 2,
            np.finfo(np.float32).max])

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-high, high)

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        if self.task == "balance":
            self.state[1] += np.pi
        return np.array(self.state)

    def is_done(self):
        x, theta = self.state[:2]
        if self.task == "balance":
            done =  x < -self.x_threshold \
                    or x > self.x_threshold \
                    or theta < np.pi - self.theta_threshold_radians \
                    or theta > np.pi + self.theta_threshold_radians
        else:
            pass
        # return bool(done)
        return False

    def step(self, action):
        #TODO: assert action is a scalar
        # assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))

        # get state
        x, th, x_dot, th_dot = self.state
        theta = self._unwrap_angle(th)

        # clip torque, update dynamics
        u = np.clip(action, -self.force_mag, self.force_mag)
        acc = self._accels(anp.array([x, th, x_dot, th_dot, u]))

        # integrate
        xacc, thacc = acc[0], acc[1]
        x_dot = x_dot + self.tau * xacc
        x  = x + self.tau * x_dot
        th_dot = th_dot + self.tau * thacc
        th = th + self.tau * th_dot + 0.5 * self.tau**2 * thacc

        # update state
        self._unwrap_angle(th)
        self.state = np.array([x, th, x_dot, th_dot])

        done = self.is_done()

        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            self.steps_beyond_done += 1
            reward = 0.0

        return self.state, reward, done, {}

    def _accels(self, vec):
        """
        Calculate the accelerations
        """
        force = vec[-1]
        pos = vec[:self.n_coords]
        vel = vec[self.n_coords:-1]
        state = vec[:-1]
        Minv = self._Minv(pos)
        B = self._B()
        C = self._C(state)
        G = self._G(pos)
        acc = anp.dot(Minv, anp.dot(B, force) - anp.dot(C, vel.reshape((2,1))) - G)
        return acc.flatten()

    def _F(self, vec):
        """
        Return derivative of state-space vector
        """
        qd = vec[self.n_coords:-1]
        qdd = self._accels(vec)
        return anp.array(list(qd) + list(qdd))

    def _M(self, pos):
        """
        Inertial Mass matrix
        """
        x, th = pos
        I = self.inertial
        d0 = self.total_mass
        d1 = self.polemass_length * anp.cos(th)

        mass_matrix = anp.array([[d0, d1],
                               [d1, I]])
        return mass_matrix

    def _C(self, state):
        """
        Coriolis matrix
        """
        x, theta, xdot, thetadot = state
        d1 = self.polemass_length * thetadot * anp.sin(theta)
        return anp.array([
                        [0, -d1],
                        [0, 0]
                        ])

    def _G(self, pos):
        """
        Gravitional matrix
        """
        x, theta = pos
        d3 = self.polemass_length * anp.sin(theta) * self.g
        return anp.array([[0], 
                        [d3]])

    def _B(self):
        """
        Force matrix
        """
        return anp.array([[1], [0]])

    def _jacobian(self):
        """
        Return the Jacobian of the full state equation
        """
        return jacobian(self._F)

    def _linearize(self, vec):
        """
        Linearize the dynamics by first order Taylor expansion
        """
        f0 = self._F(vec)
        arr = self.jacobian(vec)
        A = arr[:, :-1]
        B = arr[:, -1].reshape((2 * self.n_coords, 1))
        return f0, A, B

    def _Minv(self, pos):
        """
        Invert the mass matrix
        """
        return anp.linalg.inv(self._M(pos))

    def total_energy(self, state):
        pos = state[:self.n_coords]
        vel = state[self.n_coords:]
        return self.kinetic_energy(pos, vel) + self.potential_energy(pos)

    def kinetic_energy(self, pos, vel):
        return

    def potential_energy(self, pos):
        return

    def _unwrap_angle(self, theta):
        sign = (theta >=0)*1 - (theta < 0)*1
        theta = anp.abs(theta) % (2 * anp.pi)
        return sign*theta

    def integrate(self):
        """
        Integrate the equations of motion
        """
        raise NotImplementedError()

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold*2
        scale = screen_width/world_width
        carty = 100 # TOP OF CART
        polewidth = 10.0
        polelen = scale * 1.0
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l,r,t,b = -cartwidth/2, cartwidth/2, cartheight/2, -cartheight/2
            axleoffset =cartheight/4.0
            cart = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            cart.set_color(.62, .62, .62)
            self.viewer.add_geom(cart)
            l,r,t,b = -polewidth/2,polewidth/2,polelen-polewidth/2,-polewidth/2
            pole = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            pole.set_color(.88, .4, .4) 
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth/2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.26, .26, .26)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0,carty), (screen_width,carty))
            self.track.set_color(0,0,0)
            self.viewer.add_geom(self.track)

        if self.state is None: return None

        x = self.state
        cartx = x[0]*scale+screen_width/2.0 # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        # theta with respect to positive vertical (clockwise for positive theta)
        self.poletrans.set_rotation(x[1] + np.pi)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer: self.viewer.close()
