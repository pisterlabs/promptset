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

from surreal.algos.sac import SAC, SACConfig
import surreal.debug as debug
from surreal.envs import OpenAIGymEnv


class TestSACLongLearningTasks(unittest.TestCase):
    """
    Tests the SAC algo on up-to-1-day learning problems.
    """
    logging.getLogger().setLevel(logging.INFO)

    def test_sac_learning_on_space_invaders(self):
        # Create an Env object.
        env = OpenAIGymEnv(
            "SpaceInvaders-v4", actors=64, fire_after_reset=False, episodic_life=True, max_num_noops_after_reset=6,
            frame_skip=(2, 5)
        )

        # Create a DQN2015Config.
        config = SACConfig.make(
            "{}/../configs/sac_space_invaders_learning.json".format(os.path.dirname(__file__)),
            state_space=env.actors[0].state_space,
            action_space=env.actors[0].action_space,
            summaries=["Ls_critic[0]", "Ls_critic[1]", "L_actor", "L_alpha", "alpha", ("actions", "a_soft.value[0]")]
        )
        # Create an Algo object.
        algo = SAC(config=config, name="my-sac")

        # Point actor(s) to the algo.
        env.point_all_actors_to_algo(algo)

        # Run and wait for env to complete.
        env.run(actor_time_steps=20000000, sync=True, render=debug.RenderEnvInLearningTests)

        # Check last n episode returns.
        n = 10
        mean_last_10 = np.mean(env.historic_episodes_returns[-n:])
        print("Avg return over last 10 episodes: {}".format(mean_last_10))
        self.assertTrue(mean_last_10 > 150.0)

        env.terminate()

    def test_sac_learning_on_breakout(self):
        # Create an Env object.
        env = OpenAIGymEnv(
            "Breakout-v4", actors=128, fire_after_reset=True, episodic_life=True, max_num_noops_after_reset=6,
            frame_skip=(2, 5)
        )

        # Create a DQN2015Config.
        config = SACConfig.make(
            "{}/../configs/sac_breakout_learning.json".format(os.path.dirname(__file__)),
            state_space=env.actors[0].state_space,
            action_space=env.actors[0].action_space,
            summaries=[
                "Ls_critic[0]", "Ls_critic[1]", "L_actor", "L_alpha", "alpha", ("actions", "a_soft.value[0]"),
                "log_pi", "entropy_error_term", "log_alpha",  # TEST
                "episode.return", "episode.time_steps",
            ]
        )
        # Create an Algo object.
        algo = SAC(config=config, name="my-sac")

        # Point actor(s) to the algo.
        env.point_all_actors_to_algo(algo)

        # Run and wait for env to complete.
        env.run(actor_time_steps=20000000, sync=True, render=debug.RenderEnvInLearningTests)

        # Check last n episode returns.
        n = 10
        mean_last_10 = np.mean(env.historic_episodes_returns[-n:])
        print("Avg return over last 10 episodes: {}".format(mean_last_10))
        self.assertTrue(mean_last_10 > 200.0)

        env.terminate()
