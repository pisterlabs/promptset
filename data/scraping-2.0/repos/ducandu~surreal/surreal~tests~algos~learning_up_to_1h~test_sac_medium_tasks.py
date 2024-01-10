# Copyright 2019 ducandu GmbH. All Rights Reserved.
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

from surreal.algos.sac import SAC, SACConfig
import surreal.debug as debug
from surreal.envs import OpenAIGymEnv


class TestSACMediumLearningTasks(unittest.TestCase):
    """
    Tests the SAC algo on up-to-1-hour learning problems.
    """
    logging.getLogger().setLevel(logging.INFO)

    def test_sac_learning_on_pendulum(self):
        # Create an Env object.
        env = OpenAIGymEnv("Pendulum-v0", actors=1)

        # Create a Config.
        config = SACConfig.make(
            "{}/../configs/sac_pendulum_learning.json".format(os.path.dirname(__file__)),
            state_space=env.actors[0].state_space,
            action_space=env.actors[0].action_space
        )

        # Create an Algo object.
        algo = SAC(config=config, name="my-sac")

        # Point actor(s) to the algo.
        env.point_all_actors_to_algo(algo)

        # Run and wait for env to complete.
        env.run(ticks=9000, sync=True, render=debug.RenderEnvInLearningTests)

        # Check last n episode returns.
        n = 10
        mean_last_n = np.mean(env.historic_episodes_returns[-n:])
        print("Avg return over last n episodes: {}".format(n, mean_last_n))
        self.assertTrue(mean_last_n >= -300.0)

        env.terminate()

    def test_sac_learning_on_pendulum_with_n_step(self):
        # Create an Env object with larger episode len (which will result in more negative returns per episode).
        env = OpenAIGymEnv("Pendulum-v0", actors=2, max_episode_steps=1000)

        # Create a Config.
        config = SACConfig.make(
            "{}/../configs/sac_pendulum_learning_with_nstep.json".format(os.path.dirname(__file__)),
            state_space=env.actors[0].state_space,
            action_space=env.actors[0].action_space
        )

        # Create an Algo object.
        algo = SAC(config=config, name="my-sac")

        # Point actor(s) to the algo.
        env.point_all_actors_to_algo(algo)

        # Run and wait for env to complete.
        env.run(ticks=9000, sync=True, render=debug.RenderEnvInLearningTests)

        # Check last n episode returns.
        n = 6
        mean_last_n = np.mean(env.historic_episodes_returns[-n:])
        print("Avg return over last {} episodes: {}".format(n, mean_last_n))
        self.assertTrue(mean_last_n >= -500.0)

        env.terminate()

    def test_sac_learning_on_lunar_lander_with_n_actors(self):
        # Create an Env object.
        env = OpenAIGymEnv("LunarLander-v2", actors=4)

        # Create a Config.
        config = SACConfig.make(
            "{}/../configs/sac_lunar_lander.json".format(os.path.dirname(__file__)),
            state_space=env.actors[0].state_space,
            action_space=env.actors[0].action_space,
            summaries=[
                "episode.return", "episode.time_steps",  # TODO: "episode.duration",
                ("actions", "a_soft.value[0]"), "Ls_critic[0]", "Ls_critic[1]", "L_actor", "L_alpha", "alpha"
            ]
        )

        # Create an Algo object.
        algo = SAC(config=config, name="my-sac")

        # Point actor(s) to the algo.
        env.point_all_actors_to_algo(algo)

        # Run and wait for env to complete.
        env.run(ticks=20000, sync=True, render=debug.RenderEnvInLearningTests)

        # Check last n episode returns.
        n = 5
        mean_last_n = np.mean(env.historic_episodes_returns[-n:])
        print("Avg return over last {} episodes: {}".format(n, mean_last_n))
        self.assertTrue(mean_last_n >= 175.0)

        env.terminate()
