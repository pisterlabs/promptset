"""
Program Optimisation Environment for Q-Learning
modified from CartPoleEnv.py from OpenAI Gym 
Author: Matthew Tang
Author: Xiao Zheng
"""

import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import time
import subprocess


class ProgOptEnv(gym.Env):
    """
    Description:
        The source code of a program is provided and will be modified by a set
        of local changes. The goal is to reduce the runtime of the program while
        maintaining its correctness.

    Source:
        NIL

    Observation:
        Type: Box(4)
        Num     Observation               Min                     Max
        0       line pos                  0                       100
        1       Syntax error              0                       10.0
        2       Runtime(proportional)     0                       2.0
        3       Failure                   0                       10.0

    Actions:
        Type: Discrete(2)
        Num   Action
        0     Skip the change 
        1     Take the change

    Reward:
        Reward is 1.0 for every step taken, including the termination step

    Starting State:
        All observations are assigned a uniform random value in [-0.05..0.05]

    Episode Termination:
        There is any syntax error.
        It fails any test cases.
        Episode length is greater than 100.
        The runtime is 150% time more than a baseline.
        Solved Requirements:
        To be concluded
    """

    # metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 50}

    def __init__(self):
        self.line_pos = 1
        self.max_n_error = 0
        self.max_n_failed_cases = 0
        self.runtime_baseline = self.run_and_time('./Register_B-Baseline')
        self.runtime_threshold = 1.5  # 150% than the baseline

        self.action_space = spaces.Discrete(2)

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = 0

    def run_and_time(self, cmd):
        t = time.perf_counter()  # start time
        x = subprocess.call(cmd, shell=True)  # just run with all outputs to stdout
        t = time.perf_counter() - t  # end time
        # print(f"runtime={t:02f}")
        # assert x == 0
        return t

    def compile_source(self, cmd):
        # t = time.perf_counter()         # start time
        x = subprocess.call(cmd, shell=True)  # just run with all outputs to stdout
        # t = time.perf_counter() - t     # end time
        # print(f"runtime={t:02f}")
        return x  # return value

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        err_msg = f'{action} ({type(action)}) invalid'
        assert self.action_space.contains(action), err_msg

        self.line_pos += 1  # step forward

        # modify the program source code
        print("modify source code")

        # compile -> retrieve syntax error(s)
        # it is a simple Boolean test for now
        n_error = self.compile_source('gcc test.c -o test')
        print(f"# syntax error = {n_error}")

        # initialisations
        runtime = 1000 * self.runtime_baseline
        n_failed = 1000

        # execute -> measure runtime
        if n_error == 0:
            runtime = self.run_and_time('./Register_B > out1')  # ignore stdout # print result to out1
            print(f"updated runtime = {runtime}")

            # check against test cases (if any)
            if False:
                print(f"# failed testcase = {n_failed}")

        runtime_normalised = runtime / self.runtime_baseline  # return a normalised value
        self.state = (self.line_pos, n_error, runtime_normalised, n_failed)

        done = bool(
            n_error > self.max_n_error
            or runtime > self.runtime_baseline * self.runtime_threshold
            or n_failed > self.max_n_failed_cases
        )

        # calculate reward and return
        if not self.steps_beyond_done == 0:
            if not done:
                reward = 1.0
            # elif self.steps_beyond_done is None:
                # The change has ruined the program already!
                # self.steps_beyond_done = 0
                # reward = 1.0
            else:
                self.steps_beyond_done += 1
                reward = 0.0
        else:
            logger.warn(
                "You are calling 'step()' even though this "
                "environment has already returned done = True. You "
                "should always call 'reset()' once you receive 'done = "
                "True' -- any further steps are undefined behavior."
            )
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.state, dtype=np.float32), reward, done, {}

    def reset(self):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = 0
        return np.array(self.state, dtype=np.float32)

    def render(self, mode="human"):
        print("Warning: render() is not yet implemented")
        # if self.state is None:
        #    return None

        return None

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
