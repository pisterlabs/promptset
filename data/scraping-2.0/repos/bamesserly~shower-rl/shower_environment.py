import gymnasium as gym
import numpy as np
from gymnasium import spaces
import random

shower_length = 60
temp_target_max = 40
temp_target_min = 36
temp_target = 38
initial_temp_variation = 10
initial_temp = temp_target + random.randint(
    -1 * initial_temp_variation, initial_temp_variation
)


# No need for this class to inherit from openai gym Env.
# Just doing it for future compatibility with openai gym ML
class Shower(gym.Env):
    def __init__(self):
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Discrete(75)
        self.state = initial_temp
        self.shower_time = 0

        # self.observation_space = spaces.Box(low=np.array([0],dtype=np.float32), high=np.array([2],dtype=np.float32))

    def step(self, action):
        # increment shower_time
        self.shower_time += 1

        # determine temp change based on action
        temp_change = action - 1

        # physics bounds
        if self.state > 73:
            temp_change = -1
        if self.state <= 1:
            temp_change = 1

        # Update temperature and state
        self.state += temp_change

        # assign step reward
        reward = 0
        if temp_target_min <= self.state <= temp_target_max:
            reward = 1
        else:
            # step_reward = -1*(abs(temp_target-temp))
            reward = -1

        # check terminated
        terminated = self.shower_time == shower_length

        # print(f"  t = {self.shower_time} temp_i = {self.state - temp_change} action = {action} dt = {temp_change} temp_f = {self.state} reward = {reward} terminated = {terminated}")

        info = {}
        truncated = False

        return self.state, reward, terminated, truncated, info

    def render(self):
        pass

    def reset(self, seed=None, options=None):
        # reset initial temp, state
        self.state = temp_target + random.randint(
            -1 * initial_temp_variation, initial_temp_variation
        )

        # reset shower time
        self.shower_time = 0

        return (self.state, {})
