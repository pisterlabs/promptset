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

from surreal.algos.sac import SAC, SACConfig
from surreal.components.preprocessors.preprocessor import Preprocessor
import surreal.debug as debug
from surreal.envs import OpenAIGymEnv, GridWorld
from surreal.tests.test_util import check
from surreal.utils.numpy import one_hot


class TestSACShortLearningTasks(unittest.TestCase):
    """
    Tests the SAC algo on shorter-than-1min learning problems.
    """
    logging.getLogger().setLevel(logging.INFO)

    def test_sac_learning_on_grid_world_2x2(self):
        # Create an Env object.
        env = GridWorld("2x2", actors=1)

        # Add the preprocessor (not really necessary, as NN will automatically one-hot, but faster as states
        # are then stored in memory already preprocessed and won't have to be preprocessed again for batch-updates).
        preprocessor = Preprocessor(
            lambda inputs_: tf.one_hot(inputs_, depth=env.actors[0].state_space.num_categories)
        )

        # Create a Config.
        config = SACConfig.make(
            "{}/../configs/sac_grid_world_2x2_learning.json".format(os.path.dirname(__file__)),
            preprocessor=preprocessor,
            state_space=env.actors[0].state_space,
            action_space=env.actors[0].action_space,
            summaries=[
                "Ls_critic[0]", "L_actor", "L_alpha", "alpha",
                ("Q(0,^)", "Q[0]({'s': np.array([[1., 0., 0., 0.]]), 'a': np.array([0])})"),
                ("Q(0,->)", "Q[0]({'s': np.array([[1., 0., 0., 0.]]), 'a': np.array([1])})"),
                ("Q(0,v)", "Q[0]({'s': np.array([[1., 0., 0., 0.]]), 'a': np.array([2])})"),
                ("Q(0,<-)", "Q[0]({'s': np.array([[1., 0., 0., 0.]]), 'a': np.array([3])})"),
                ("Q(1,->)", "Q[0]({'s': np.array([[0., 1., 0., 0.]]), 'a': np.array([1])})")
            ]
        )

        # Create an Algo object.
        algo = SAC(config=config, name="my-sac")

        # Point actor(s) to the algo.
        env.point_all_actors_to_algo(algo)

        # Run and wait for env to complete.
        env.run(ticks=700, sync=True, render=debug.RenderEnvInLearningTests)

        # Check learnt Q-function.
        q = algo.Q[0](dict(s=one_hot(np.array([0, 0, 0, 0, 1, 1, 1, 1]), depth=4), a=np.array([0, 1, 2, 3, 0, 1, 2, 3])))
        print(q)
        self.assertTrue(q[1] < min(q[2:]) and q[1] < q[0])  # q(s=0,a=right) is the worst
        check(q[5], 1.0, decimals=1)  # Q(1,->) is close to 1.0.
        #check(q, [0.8, -5.0, 0.9, 0.8, 0.8, 1.0, 0.9, 0.9], decimals=1)  # a=up,down,left,right

        # Check last n episode returns.
        n = 10
        mean_last_n = np.mean(env.historic_episodes_returns[-n:])
        print("Avg return over last {} episodes: {}".format(n, mean_last_n))
        self.assertTrue(mean_last_n >= 0.7)

        env.terminate()

    def test_sac_learning_on_cart_pole_with_n_actors(self):
        # Create an Env object.
        env = OpenAIGymEnv("CartPole-v0", actors=2)

        # Create a Config.
        config = SACConfig.make(
            "{}/../configs/sac_cart_pole_learning_n_actors.json".format(os.path.dirname(__file__)),
            state_space=env.actors[0].state_space,
            action_space=env.actors[0].action_space
        )

        # Create an Algo object.
        algo = SAC(config=config, name="my-sac")

        # Point actor(s) to the algo.
        env.point_all_actors_to_algo(algo)

        # Run and wait for env to complete.
        env.run(ticks=2000, sync=True, render=debug.RenderEnvInLearningTests)

        # Check last n episode returns.
        last_n = 4
        mean_last_episodes = np.mean(env.historic_episodes_returns[-last_n:])
        print("Avg return over last {} episodes: {}".format(last_n, mean_last_episodes))
        self.assertTrue(mean_last_episodes > 160.0)

        env.terminate()
