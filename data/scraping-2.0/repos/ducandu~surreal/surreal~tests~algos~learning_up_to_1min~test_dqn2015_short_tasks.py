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
import tensorflow as tf
import unittest

from surreal.algos.dqn2015 import DQN2015, DQN2015Config
from surreal.components import Preprocessor
import surreal.debug as debug
from surreal.envs import OpenAIGymEnv, GridWorld
from surreal.tests.test_util import check


class TestDQN2015ShortLearningTasks(unittest.TestCase):
    """
    Tests the DQN2015 algo on shorter-than-1min learning problems.
    """
    logging.getLogger().setLevel(logging.INFO)

    def test_dqn2015_learning_on_grid_world(self):
        # Create an Env object.
        env = GridWorld("2x2")

        # Add the preprocessor.
        preprocessor = Preprocessor(
            lambda inputs_: tf.one_hot(inputs_, depth=env.actors[0].state_space.num_categories)
        )
        # Create a Config.
        config = DQN2015Config.make(  # type: DQN2015Config
            "{}/../configs/dqn2015_grid_world_2x2_learning.json".format(os.path.dirname(__file__)),
            preprocessor=preprocessor,
            state_space=env.actors[0].state_space,
            action_space=env.actors[0].action_space
        )

        # Create an Algo object.
        algo = DQN2015(config=config, name="my-dqn")

        # Point actor(s) to the algo.
        env.point_all_actors_to_algo(algo)

        # Run and wait for env to complete.
        env.run(ticks=3000, sync=True, render=debug.RenderEnvInLearningTests)

        # Check last n episode returns.
        n = 10
        mean_last_n = np.mean(env.historic_episodes_returns[-n:])
        print("Avg return over last {} episodes: {}".format(n, mean_last_n))
        self.assertTrue(mean_last_n >= 0.3)

        # Check learnt Q-function.
        check(algo.Q(
            np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        ), [[0.8, -5.0, 0.9, 0.8], [0.8, 1.0, 0.9, 0.9]], decimals=1)  # a=up,down,left,right

        env.terminate()

    def test_dqn2015_learning_on_4x4_grid_world_with_n_actors(self):
        # Create an Env object.
        env = GridWorld("4x4", actors=8)

        # Add the preprocessor.
        preprocessor = Preprocessor(
            lambda inputs_: tf.one_hot(inputs_, depth=env.actors[0].state_space.num_categories)
        )

        # Create a Config.
        config = DQN2015Config.make(  # type: DQN2015Config
            "{}/../configs/dqn2015_grid_world_4x4_learning_n_actors.json".format(os.path.dirname(__file__)),
            preprocessor=preprocessor,
            state_space=env.actors[0].state_space,
            action_space=env.actors[0].action_space
        )

        # Create an Algo object.
        algo = DQN2015(config=config, name="my-dqn")

        # Point actor(s) to the algo.
        env.point_all_actors_to_algo(algo)

        # Run and wait for env to complete.
        env.run(ticks=4000, sync=True, render=debug.RenderEnvInLearningTests)

        # Check last n episode returns.
        n = 10
        mean_last_n = np.mean(env.historic_episodes_returns[-n:])
        print("Avg return over last {} episodes: {}".format(n, mean_last_n))
        self.assertTrue(mean_last_n >= -0.4)

        # Check learnt Q-function for states 0 and 1, action=down (should be larger 0.0, ideally 0.5).
        action_values = algo.Q(preprocessor(np.array([0, 1])))
        self.assertTrue(action_values[0][2] >= 0.0)
        self.assertTrue(action_values[1][2] >= 0.0)

        env.terminate()

    def test_dqn2015_learning_on_cart_pole_with_n_actors(self):
        # Create an Env object.
        env = OpenAIGymEnv("CartPole-v0", actors=4, num_cores=None)

        # Create a Config.
        config = DQN2015Config.make(  # type: DQN2015Config
            "{}/../configs/dqn2015_cart_pole_learning_n_actors.json".format(os.path.dirname(__file__)),
            state_space=env.actors[0].state_space,
            action_space=env.actors[0].action_space
        )

        # Create an Algo object.
        algo = DQN2015(config=config, name="my-dqn")

        # Point actor(s) to the algo.
        env.point_all_actors_to_algo(algo)

        # Run and wait for env to complete.
        env.run(ticks=3000, sync=True, render=debug.RenderEnvInLearningTests)

        # Check last n episode returns.
        n = 10
        mean_last_n = np.mean(env.historic_episodes_returns[-n:])
        print("Avg return over last {} episodes: {}".format(n, mean_last_n))
        self.assertTrue(mean_last_n > 130.0)

        env.terminate()
