#!/usr/bin/env python3
""" NOT A REAL TRAINING SCRIPT
Please check the README.md in located in this same folder
for an explanation of this script"""

import gym
from gym.spaces import MultiDiscrete
import numpy
import time
from openai_ros2.envs import LobotArmEnv
import rclpy
from typing import Type


def main(args=None):
    env: LobotArmEnv = gym.make('LobotArmDiscrete-v1') # Discrete with noise
    env.reset()
    action_space: Type[MultiDiscrete] = env.action_space
    rclpy.spin_once(env.node)
    while True:
        print("-------------Starting----------------")
        for x in range(500):
            # action = numpy.array([1, 0, 4])
            action = action_space.sample()
            action_do_nothing = numpy.array([2, 2, 2])
            observation, reward, done, info = env.step(action_do_nothing)
            # Type hints
            observation: numpy.ndarray
            reward: float
            done: bool
            info: dict
            if done:
                break
        time.sleep(0.8)
        print("-------------Resetting environment---------------")
        env.reset()
        print("-------------Reset finished----------------")


if __name__ == '__main__':
    main()
