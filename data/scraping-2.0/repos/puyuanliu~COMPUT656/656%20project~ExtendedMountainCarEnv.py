# -----------------------------------------------------------
# Enviroment for Extended Mountain Car problem.
# Extended Mountain Car problem has a different goal compared to the usual
# Mountain Car problem, that is, stop at a target height instead of getting
# out of the bottom.
# Extended Mountain Car environment is modified based on the Mountain Car
# environment of Gym package from OpenAI.
#
# (C) 2020 Puyuan Liu, Department of Computing Science, University of Alberta
# Released under GNU Public License (GPL)
# email puyuan@ualberta.ca
# -----------------------------------------------------------

import gym
import time
import math
import itertools
from math import floor, log
import numpy as np
from gym import spaces
from gym.utils import seeding
import pyglet

class ExtendedMountainCarEnv(gym.Env):
    """
    Description:
        The agent (a car) is started at the bottom of a valley. For any given
        state the agent may choose to accelerate to the left, right or cease
        any acceleration.
    Source:
        The environment appeared first in Andrew Moore's PhD Thesis (1990).
    Observation:
        Type: Box(2)
        Num    Observation               Min            Max
        0      Car Position              -1.2           0.6
        1      Car Velocity              -0.07          0.07
    Actions:
        Type: Discrete(3)
        Num    Action
        0      Accelerate to the Left
        1      Don't accelerate
        2      Accelerate to the Right
        Note: This does not affect the amount of velocity affected by the
        gravitational pull acting on the car.
    Reward:
         Reward of 0 is awarded if the agent reached the flag (position = 0.5)
         on top of the mountain.
         Reward of -1 is awarded if the position of the agent is less than 0.5.
    Starting State:
         The position of the car is assigned a uniform random value in
         [-0.6 , -0.4].
         The starting velocity of the car is always assigned to 0.
    Episode Termination:
         The car position is more than 0.5
         Episode length is greater than 200
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 60
    }

    def __init__(self, goal_velocity=0, goal_height=0.1, init_pos=1, height_criteria=0.01, velocity_criteria = 0.001):
        # goal height: the desired height that we want to reach
        # init_pos: 0 means some height below the desired height
        #           1 means some height above the desired height
        # velocity_criteria: the maximum magnitude of the car's velocity when
        #                    reaching the desired height to end the episode.
        self.min_position = -1.4
        self.max_position = 0.4
        self.max_speed = 0.07
        self.goal_position = goal_height
        self.goal_velocity = goal_velocity
        self.init_pos = init_pos
        self.height_creteria = height_criteria
        self.velocity_criteria = velocity_criteria
        self.force = 0.00005
        self.gravity = 0.00015

        self.low = np.array(
            [self.min_position, -self.max_speed], dtype=np.float32
        )
        self.high = np.array(
            [self.max_position, self.max_speed], dtype=np.float32
        )

        self.viewer = None

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            self.low, self.high, dtype=np.float32
        )

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        position, velocity = self.state
        velocity += (action - 1) * self.force + math.cos(3 * position) * (-self.gravity)
        velocity = np.clip(velocity, -self.max_speed, self.max_speed)
        position += velocity
        position = np.clip(position, self.min_position, self.max_position)
        if (position == self.min_position and velocity < 0):
            velocity = 0
        if (position == self.max_position and velocity > 0):
            velocity = 0
        done = bool(
            abs(position-self.goal_position) < self.height_creteria and abs(velocity-self.goal_velocity)<self.velocity_criteria
        )
        reward = -1.0

        self.state = (position, velocity)
        return np.array(self.state), reward, done, {}

    def reset(self):
        self.state = np.array([self.np_random.uniform(low=self.min_position, high=self.max_position), 0])
        return np.array(self.state)

    def _height(self, xs):
        return np.sin(3 * xs) * .45 + .55

    def render(self, mode='human'):
        screen_width = 1200
        screen_height = 800

        world_width = self.max_position - self.min_position
        scale = screen_width / world_width
        carwidth = 40
        carheight = 20

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            xs = np.linspace(self.min_position, self.max_position, 100)
            ys = self._height(xs)
            xys = list(zip((xs - self.min_position) * scale, ys * scale))

            self.track = rendering.make_polyline(xys)
            self.track.set_linewidth(4)
            self.viewer.add_geom(self.track)

            clearance = 10

            l, r, t, b = -carwidth / 2, carwidth / 2, carheight, 0
            car = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            car.add_attr(rendering.Transform(translation=(0, clearance)))
            self.cartrans = rendering.Transform()
            car.add_attr(self.cartrans)
            self.viewer.add_geom(car)
            frontwheel = rendering.make_circle(carheight / 2.5)
            frontwheel.set_color(.5, .5, .5)
            frontwheel.add_attr(
                rendering.Transform(translation=(carwidth / 4, clearance))
            )
            frontwheel.add_attr(self.cartrans)
            self.viewer.add_geom(frontwheel)
            backwheel = rendering.make_circle(carheight / 2.5)
            backwheel.add_attr(
                rendering.Transform(translation=(-carwidth / 4, clearance))
            )
            backwheel.add_attr(self.cartrans)
            backwheel.set_color(.5, .5, .5)
            self.viewer.add_geom(backwheel)
            flagx = (self.goal_position-self.min_position) * scale
            flagy1 = self._height(self.goal_position) * scale
            flagy2 = flagy1 + 50
            flagpole = rendering.Line((flagx, flagy1), (flagx, flagy2))
            self.viewer.add_geom(flagpole)
            flag = rendering.FilledPolygon(
                [(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)]
            )
            flag.set_color(.8, .8, 0)
            self.viewer.add_geom(flag)
        pos = self.state[0]
        self.cartrans.set_translation(
            (pos-self.min_position) * scale, self._height(pos) * scale
        )
        self.cartrans.set_rotation(math.cos(3 * pos))
        if self.state[1] < 0:
            # If the car is pointing left, we use green color for the text
            current_color = (0, 204, 102, 255)
        else:
            # If the car is pointing right, we use red color for the text
            current_color = (255, 0, 0, 255)
        if abs(self.state[0]-self.goal_position) <= 20*self.height_creteria:
            text1 = "Current velocity is: %.3f" % self.state[1]
            text2 = "Our criteria is %.3f" % self.velocity_criteria
            label = pyglet.text.Label(text1, font_size=25,
                                          x=600, y=700, anchor_x='left', anchor_y='bottom',
                                          color=current_color)
            self.viewer.add_onetime(DrawText(label))
            label = pyglet.text.Label(text2, font_size=25,
                                          x=600, y=660, anchor_x='left', anchor_y='bottom',
                                          color=current_color)
            self.viewer.add_onetime(DrawText(label))
            #self.viewer.render(return_rgb_array=False)
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def get_keys_to_action(self):
        # Control with left and right arrow keys.
        return {(): 1, (276,): 0, (275,): 2, (275, 276): 1}

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

class DrawText:
    def __init__(self, label:pyglet.text.Label):
        self.label=label
    def render(self):
        self.label.draw()