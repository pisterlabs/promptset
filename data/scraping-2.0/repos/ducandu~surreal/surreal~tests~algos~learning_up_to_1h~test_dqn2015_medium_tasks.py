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

from surreal.algos.dqn2015 import DQN2015, DQN2015Config
import surreal.debug as debug
from surreal.envs import OpenAIGymEnv


class TestDQN2015MediumLearningTasks(unittest.TestCase):
    """
    Tests the DQN2015 algo on up-to-1-hour learning problems.
    """
    logging.getLogger().setLevel(logging.INFO)

    def test_dqn2015_learning_on_lunar_lander_with_8_actors(self):
        # Create an Env object.
        env = OpenAIGymEnv("LunarLander-v2", actors=8)

        # Create a DQN2015Config.
        dqn_config = DQN2015Config.make(  # type: DQN2015Config
            "{}/../configs/dqn2015_lunar_lander_learning_n_actors.json".format(os.path.dirname(__file__)),
            state_space=env.actors[0].state_space,
            action_space=env.actors[0].action_space
        )

        # Create an Algo object.
        algo = DQN2015(config=dqn_config, name="my-dqn")

        # Point actor(s) to the algo.
        env.point_all_actors_to_algo(algo)

        # Run and wait for env to complete.
        env.run(ticks=50000, sync=True, render=debug.RenderEnvInLearningTests)

        # Check last n episode returns.
        n = 10
        mean_last_n = np.mean(env.historic_episodes_returns[-n:])
        print("Avg return over last {} episodes: {}".format(n, mean_last_n))
        self.assertTrue(mean_last_n > 100.0)

        env.terminate()
