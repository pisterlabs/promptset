# Copyright 2019 ducandu GmbH, All Rights Reserved
# (this is a modified version of the Apache 2.0 licensed RLgraph file of the same name).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import logging
import numpy as np
import os
import unittest

from surreal.algos.dddqn import DDDQN, DDDQNConfig
import surreal.debug as debug
from surreal.envs import OpenAIGymEnv


class TestDDDQNMediumLearningTasks(unittest.TestCase):
    """
    Tests the DDDQN algo on up-to-1-hour learning problems.
    """
    logging.getLogger().setLevel(logging.INFO)

    def test_dddqn_learning_on_mountain_car_4_actors(self):
        # Note: MountainCar is tricky as per its reward function: Hence, we need a quite large episode
        # cutoff to solve it with ease.
        # With a large enough n-step, the algo should be able to learn the env very quickly after having solved
        # it once via randomness.
        env = OpenAIGymEnv("MountainCar-v0", actors=4, max_episode_steps=5000)

        # Create a DQN2015Config.
        dqn_config = DDDQNConfig.make(
            "{}/../configs/dddqn_mountain_car_learning_n_actors.json".format(os.path.dirname(__file__)),  # TODO: filename wrong (num actors)
            state_space=env.actors[0].state_space,
            action_space=env.actors[0].action_space
        )

        # Create an Algo object.
        algo = DDDQN(config=dqn_config, name="my-dqn")

        # Point actor(s) to the algo.
        env.point_all_actors_to_algo(algo)

        # Run and wait for env to complete.
        env.run(ticks=7000, sync=True, render=debug.RenderEnvInLearningTests)

        # Check last n episode returns.
        last_n = 10
        mean_last_episodes = np.mean(env.historic_episodes_returns[-last_n:])
        print("Avg return over last {} episodes: {}".format(last_n, mean_last_episodes))
        self.assertTrue(mean_last_episodes > -200.0)

        env.terminate()

    def test_dddqn_learning_on_lunar_lander_with_4_actors(self):
        # Create an Env object.
        env = OpenAIGymEnv("LunarLander-v2", actors=4)

        # Create a DQN2015Config.
        config = DDDQNConfig.make(
            "{}/../configs/dddqn_lunar_lander_learning_n_actors.json".format(os.path.dirname(__file__)),
            state_space=env.actors[0].state_space,
            action_space=env.actors[0].action_space
        )

        # Create an Algo object.
        algo = DDDQN(config=config, name="my-dddqn")

        # Point actor(s) to the algo.
        env.point_all_actors_to_algo(algo)

        # Run and wait for env to complete.
        env.run(ticks=30000, sync=True, render=debug.RenderEnvInLearningTests)

        # Check last n episode returns.
        n = 10
        mean_last_n = np.mean(env.historic_episodes_returns[-n:])
        print("Avg return over last {} episodes: {}".format(n, mean_last_n))
        self.assertTrue(mean_last_n > 150.0)

        env.terminate()
