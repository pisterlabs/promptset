import time
from typing import Tuple

import gym
import numpy as np
import random
import pygame
from pygame import gfxdraw

# physics constants
GRAVITY = 1.62  # [m/s^2] moon gravity
UP_ACCELERATION = 2.5  # [m/s^2] acc of thruster upwards
SIDE_ACCELERATION = 2  # [m/s^2]  acc of thruster sideways

PLAYGROUND_WIDTH = 200  # [m]
PLAYGROUND_HEIGHT = 100  # [m]
LANDING_PAD_WIDTH = 30

CRASH_THRESHOLD_X = 2  # [m/s] terminal sideways velocity on impact, which would make spacecraft fall over
CRASH_THRESHOLD_Y = 6  # [m/s] terminal velocity on impact
START_OFFSET_X = 10  # this outer region cannot be x start position
START_OFFSET_Y = 20  # start this offset lower than max height

# render constants
SCREEN_WIDTH = 750
SCREEN_HEIGHT = 500
EXPECTED_FPS = 100

OOB_REWARD = -50  # reward for out of bounds
SUCCESS_REWARD = 210
PAD_DISTANCE_REWARD = -3  # factor for landing closer to center
LONG_TIME_REWARD = -50  # reward if max time is elapsed
MAX_PLAY_TIME = 50

# Additional Rewards not used because it made the environment too difficult
CRASH_REWARD_X = 0  # reward for crashing by having too high sideways velocity on impact
CRASH_REWARD_Y = 0  # reward for crashing by falling to fast
SMOOTH_LANDING_FACTOR = 0  # Factor for rewarding smooth landing
BOOSTER_REWARD_FACTOR = 0  # Factor for constant negative reward while playing

class CustomLunarLander(gym.Env):
    """
    Land a Lunar Lander on a landing pad on the moon. Variation of LunarLander-v2 from OpenAI-Gym:
    https://github.com/openai/gym/blob/master/gym/envs/box2d/lunar_lander.py

    The task is to land on a landing pad at position [0, 0] with a low falling velocity, while the lunar lander
    is starting at a random x-position in the air.

    Observation Space:
    The observation space has entries in the following order:
    0. x position in [-PLAYGROUND_WIDTH / 2, PLAYGROUND_WIDTH / 2]
    1. y position in [0, PLAYGROUND_HEIGHT]
    2. x velocity in [-inf, +inf], where right is positive direction
    3. y velocity in [-inf, +inf], where up is positive direction

    Action Space:
    There are the following possible actions (all integer):
    0. Do nothing
    1. Thrust Left
    2. Thrust Up
    3. Thrust Right

    Reward Space:
    The only positive reward is obtained by hitting the landing pad. There is negative reward for crashing due to
    out of bounds, crashing due to hitting ground with too high x or y velocity (set by threshold).
    Additionally, a smooth negative reward proportional to the terminal x and y velocity is present.
    A negative reward for using booster incentivizes the agent to finish fast.
    """

    def __init__(self, step_size: float):
        # gym
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(4,), dtype=np.float32)
        self.reward_range = (float('-inf'), 200)
        self.state = None

        self.step_size = step_size  # seconds
        self.num_steps = 0

        # GUI only
        self.screen = None
        self.saved_action = None
        self.frame = None

    def reset(self) -> np.ndarray:
        self.state = np.asarray(
            [
                random.randint(int(-PLAYGROUND_WIDTH / 2) + START_OFFSET_X,
                               int(PLAYGROUND_WIDTH / 2) - START_OFFSET_X),
                PLAYGROUND_HEIGHT - START_OFFSET_Y,
                0,
                0,
            ], dtype=float)
        self.num_steps = 0
        return self.state

    def step(self, action: int) -> Tuple[list, float, bool, dict]:

        if self.state is None:
            raise AssertionError("Environment is not reset yet.")

        self.saved_action = action

        # update velocities
        if action == 1:  # left
            self.state[2] -= SIDE_ACCELERATION * self.step_size
        elif action == 2:  # Up
            self.state[3] += UP_ACCELERATION * self.step_size
        elif action == 3:  # right
            self.state[2] += SIDE_ACCELERATION * self.step_size
        self.state[3] -= GRAVITY * self.step_size  # gravity pulls down

        # update positions
        self.state[0] += self.step_size * self.state[2]
        self.state[1] += self.step_size * self.state[3]

        # check termination conditions
        # left or right out of bounds
        if self.state[0] < -PLAYGROUND_WIDTH / 2 or self.state[0] > PLAYGROUND_WIDTH / 2:
            return self.state, OOB_REWARD, True, {}

        # Above playground
        if self.state[1] > PLAYGROUND_HEIGHT:
            return self.state, OOB_REWARD, True, {}

        # Landed
        reward = 0
        info = ''

        if action != 0:
            reward += BOOSTER_REWARD_FACTOR * self.step_size

        if self.state[1] <= 0:
            # test if landed on pad
            if -LANDING_PAD_WIDTH / 2 < self.state[0] < LANDING_PAD_WIDTH / 2:
                reward += SUCCESS_REWARD
                info += ' -success'

            # test if we crashed
            if self.state[2] > CRASH_THRESHOLD_X or self.state[2] < -CRASH_THRESHOLD_X:
                reward += CRASH_REWARD_X
                info += ' -x_crash'
            if self.state[3] < -CRASH_THRESHOLD_Y:
                reward += CRASH_REWARD_Y
                info += ' -y_crash'
            # Additional smooth reward for landing softly in x and y direction, EDIT: only y
            # reward += SMOOTH_LANDING_FACTOR * abs(self.state[2])
            reward += SMOOTH_LANDING_FACTOR * abs(self.state[3])
            reward += PAD_DISTANCE_REWARD * abs(self.state[0])
            return self.state, reward, True, {'info_str': info}

        # Test if max play time is over
        if self.num_steps * self.step_size > MAX_PLAY_TIME:
            reward += LONG_TIME_REWARD
            reward += PAD_DISTANCE_REWARD * abs(self.state[0])
            return self.state, reward, True, {}

        # Nothing happened
        self.num_steps += 1
        return self.state, reward, False, {}

    def render(self, mode: str = 'human'):
        if self.frame is None:
            pygame.init()
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

        canvas = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        canvas.fill((0, 0, 0))

        xr = SCREEN_WIDTH / PLAYGROUND_WIDTH
        yr = SCREEN_HEIGHT / PLAYGROUND_HEIGHT

        # Lunar lander
        size_on_screen = SCREEN_WIDTH / 40
        gfxdraw.filled_circle(
            canvas,
            int((self.state[0] + PLAYGROUND_WIDTH / 2) * xr),
            SCREEN_HEIGHT - int(self.state[1] * yr),
            int(size_on_screen),
            (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)),
        )

        # Exhaustion gases
        if self.saved_action != 0:

            offset_x = 0
            offset_y = 0

            if self.saved_action == 1:
                offset_x = 3 / 2 * size_on_screen
            elif self.saved_action == 2:
                offset_y = 3 / 2 * size_on_screen
            elif self.saved_action == 3:
                offset_x = -3 / 2 * size_on_screen

            gfxdraw.filled_circle(
                canvas,
                int((self.state[0] + PLAYGROUND_WIDTH / 2) * xr) + int(offset_x),
                SCREEN_HEIGHT - int(self.state[1] * yr) + int(offset_y),
                int(size_on_screen / 4),
                (255, 255, 255),
            )

        # Landing pad
        gfxdraw.filled_circle(
            canvas,
            int(LANDING_PAD_WIDTH / 2 * xr) + int(PLAYGROUND_WIDTH / 2 * xr),
            SCREEN_HEIGHT,
            int(SCREEN_WIDTH / 80),
            (255, 0, 0),
        )
        gfxdraw.filled_circle(
            canvas,
            int(-LANDING_PAD_WIDTH / 2 * xr) + int(PLAYGROUND_WIDTH / 2 * xr),
            SCREEN_HEIGHT,
            int(SCREEN_WIDTH / 80),
            (255, 0, 0),
        )

        # Finish
        self.screen.blit(canvas, (0, 0))
        pygame.display.flip()
        time.sleep(1 / EXPECTED_FPS)

    def close(self):
        if self.frame is not None:
            pygame.quit()
