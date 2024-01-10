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
from surreal.components.preprocessors import Preprocessor, GrayScale, ImageResize, ImageCrop, Sequence
import surreal.debug as debug
from surreal.envs import OpenAIGymEnv
from surreal.spaces import Int


class TestDDDQNLongLearningTasks(unittest.TestCase):
    """
    Tests the DDDQN algo on up-to-1-day learning problems.
    """
    logging.getLogger().setLevel(logging.INFO)

    def test_dddqn_learning_on_car_racing(self):
        # Action-map: Discrete to Continuous, 9 actions.
        # 0=noop
        # 1=left
        # 2=right
        # 3=break only
        # 4=break and left
        # 5=break and right
        # 6=gas only
        # 7=gas and left
        # 8=gas and right
        def action_map(a):
            b = np.reshape(a, (-1, 1))
            return np.where(
                #b == 0, [0.0, 0.0, 0.0], np.where(
                #    b == 1, [-1.0, 0.0, 0.0], np.where(
                #        b == 2, [1.0, 0.0, 0.0], np.where(
                            b == 0, [0.0, 0.0, 1.0], np.where(
                                b == 1, [-1.0, 0.0, 1.0], np.where(
                                    b == 2, [1.0, 0.0, 1.0], np.where(
                                        b == 3, [0.0, 1.0, 0.0], np.where(
                                            b == 4, [-1.0, 1.0, 0.0], [1.0, 1.0, 0.0]
            )))))

        # Create an Env object.
        env = OpenAIGymEnv("CarRacing-v0", actors=1, action_map=action_map)

        # Create a DQN2015Config.
        config = DDDQNConfig.make(
            "{}/../configs/dddqn_car_racing_learning.json".format(os.path.dirname(__file__)),
            preprocessor=Preprocessor(
                #ImageCrop(x=0, y=0, width=150, height=167),
                GrayScale(keepdims=True),
                ImageResize(width=84, height=84, interpolation="bilinear"),
                lambda inputs_: ((inputs_ / 128) - 1.0).astype(np.float32),
                # simple preprocessor: [0,255] to [-1.0,1.0]
                Sequence(sequence_length=4, adddim=False)
            ),
            state_space=env.actors[0].state_space,
            action_space=Int(6)
        )
        # Create an Algo object.
        algo = DDDQN(config=config, name="my-dddqn")

        # Point actor(s) to the algo.
        env.point_all_actors_to_algo(algo)

        # Run and wait for env to complete.
        env.run(ticks=20000, sync=True, render=debug.RenderEnvInLearningTests)

        # Check last n episode returns.
        n = 10
        mean_last_n = np.mean(env.historic_episodes_returns[-n:])
        print("Avg return over last {} episodes: {}".format(n, mean_last_n))
        self.assertTrue(mean_last_n > 150.0)

        env.terminate()

    def test_dddqn_learning_on_breakout(self):
        # Create an Env object.
        env = OpenAIGymEnv(
            "Breakout-v4", actors=16, fire_after_reset=True, episodic_life=True, max_num_noops_after_reset=8,
            frame_skip=(2, 5)
        )

        preprocessor = Preprocessor(
            ImageCrop(x=5, y=29, width=150, height=167),
            GrayScale(keepdims=True),
            ImageResize(width=84, height=84, interpolation="bilinear"),
            lambda inputs_: ((inputs_ / 128) - 1.0).astype(np.float32),  # simple preprocessor: [0,255] to [-1.0,1.0]
            Sequence(sequence_length=4, adddim=False)
        )
        # Create a DQN2015Config.
        config = DDDQNConfig.make(
            "{}/../configs/dddqn_breakout_learning.json".format(os.path.dirname(__file__)),
            preprocessor=preprocessor,
            state_space=env.actors[0].state_space,
            action_space=env.actors[0].action_space
        )
        # Create an Algo object.
        algo = DDDQN(config=config, name="my-dddqn")

        # Point actor(s) to the algo.
        env.point_all_actors_to_algo(algo)

        # Run and wait for env to complete.
        env.run(actor_time_steps=10000000, sync=True, render=debug.RenderEnvInLearningTests)

        # Check last n episode returns.
        n = 10
        mean_last_n = np.mean(env.historic_episodes_returns[-n:])
        print("Avg return over last {} episodes: {}".format(n, mean_last_n))
        self.assertTrue(mean_last_n > 150.0)

        env.terminate()
