# Copyright 2019 ducandu GmbH, All Rights Reserved
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

from collections import namedtuple
import logging
import numpy as np
import os
import tensorflow as tf
import unittest

import surreal.debug as debug
# Override debug setting. Needed for some of the tests.
debug.KeepLastMemoryBatch = True

from surreal.algos.sac import SAC, SACConfig, SACLoss
from surreal.envs import OpenAIGymEnv
from surreal.tests.test_util import check


class TestSACFunctionality(unittest.TestCase):
    """
    Tests the SAC algo functionality (loss functions, execution logic, etc.).
    """
    logging.getLogger().setLevel(logging.INFO)

    def test_sac_compilation(self):
        """
        Tests the c'tor of SAC.
        """
        env = OpenAIGymEnv("Pong-v0", actors=2)
        # Create a Config (for any Atari game).
        config = SACConfig.make(
            "{}/../configs/sac_breakout_learning.json".format(os.path.dirname(__file__)),
            memory_capacity=1000,
            state_space=env.actors[0].state_space,
            action_space=env.actors[0].action_space
        )
        sac = SAC(config)
        print("SAC built ({}).".format(sac))

        env.terminate()

    def test_sac_loss_function(self):
        # Batch of size=2.
        input_ = {
            "s": np.random.random(size=(2, 2)),  # states don't matter for this test as Q-funcs are faked.
            "a": np.array([[-0.5], [0.5]]),  # action space = Float(shape=(1,))
            "r": np.array([9.4, -1.23]),
            "t": np.array([False, False]),
            "s_": np.random.random(size=(2, 2))  # states don't matter for this test as Q-funcs are faked.
        }

        # Fake pi/q-nets. Just have to be callables, returning some q-values.
        def pi(s, log_likelihood=False):
            assert log_likelihood is True
            # Return fake action sample and log-likelihoods.
            # Actions according to action-space (Float(1,)), log-likelihoods always with shape=().
            return np.array([[-0.5], [0.5]]), np.array([-0.4, -1.0])

        pi.get_weights = lambda as_ref: []
        gamma = 1.0
        q_nets = [lambda s_a: np.array([10.0, -90.6]), lambda s_a: np.array([10.1, -90.5])]
        q_nets[0].get_weights = lambda as_ref: []
        q_nets[1].get_weights = lambda as_ref: []
        target_q_nets = [lambda s_a: np.array([12.0, -8.0]), lambda s_a: np.array([22.3, 10.5])]
        target_q_nets[0].get_weights = lambda as_ref: []
        target_q_nets[1].get_weights = lambda as_ref: []
        alpha = tf.Variable(0.5, dtype=tf.float64)
        entropy_target = 0.97

        out = SACLoss()(
            input_, alpha, entropy_target, pi, q_nets, target_q_nets,
            namedtuple("FakeSACConfig", ["gamma", "entropy_target", "optimize_alpha"])(
                gamma=gamma, entropy_target=entropy_target, optimize_alpha=True
            )
        )

        # Critic Loss.
        """
        Calculation:
        batch of 2, gamma=1.0
        a' = pi(s') = [-0.5, 0.5]
        a' lllh = [-0.4, -1.0] -> sampled a's log likelihoods
        Q1t(s'a') = [12 -8]
        Q2t(s'a') = [22.3 10.5]
        Qt(s'a') = [12 -8]  (reduce min over two Q-nets)
        Q1(s'a') = [10 -90.6]
        Q2(s'a') = [10.1 -90.5]
        Li = E(batch)| 0.5( (r + gamma (Qt(s'a') - alpha*log(pi(a'|s'))) ) - Qi(s,a))^2 |

        L1 = 0.5 * | (9.4 + (12 - 0.5*-0.4) - 10)^2 + (-1.23 + (-8 - 0.5*-1.0) - -90.6)^2 | / 2
        L1 = 0.5 * |  (11.6)^2 + (81.87)^2 | / 2
        L1 = 3418.62845 / 2
        L1 = 1709.314225

        L2 = 0.5 * | (9.4 + (12 - 0.5*-0.4) - 10.1)^2 + (-1.23 + (-8 - 0.5*-1.0) - -90.5)^2 | / 2
        L2 = 0.5 * |  (11.5)^2 + (81.77)^2 | / 2
        L2 = 3409.29145 / 2
        L2 = 1704.645725
        """
        expected_critic_loss = [np.array(1709.314225), np.array(1704.645725)]
        check([out[0][i].numpy() for i in range(2)], expected_critic_loss, decimals=3)

        # Actor loss.
        """
        Calculation:
        batch of 2, gamma=1.0
        log(pi(a|s)) = a lllh = [-0.4, -1.0]
        Q1(s,a) = [10.0, -90.6]
        Q2(s,a) = [10.1, -90.5]
        Q(s,a) = [10.0, -90.6]  <- reduce_min
        L = E(batch)| ( alpha * log(pi(a,s)) - Q(s,a)) |
        L = [(alpha * -0.4 - 10.0) + (alpha * -1.0 - -90.6)] / 2
        L = [(0.5*-0.4 - 10.0) + (0.5*-1.0 - - 90.6)] / 2
        L = (-10.2 + 90.1) / 2
        L = 39.95
        """
        expected_actor_loss = 39.95
        check(out[3].numpy(), expected_actor_loss, decimals=3)

        # Alpha loss.
        """
        Calculation:
        batch of 2, gamma=1.0
        H = entropy_target = 0.97
        log(pi(a|s)) = a lllh = [-0.4, -1.0]
        L = E(batch)| (-alpha * log(pi(a,s)) - alpha H) |

        # In the SAC-paper, α is used directly, however the implementation uses log(α).
        # See the discussion in https://github.com/rail-berkeley/softlearning/issues/37.

        L = [(-log(alpha) * -0.4 - log(alpha)*0.97) + (-log(alpha) * -1.0 - log(alpha) * 0.97)] / 2
        L = [(-log(0.5)*-0.4 - log(0.5)*0.97) + (-log(0.5)*-1.0 - log(0.5)*0.97)] / 2
        L = [(0.69315*-0.4 - -0.69315*0.97) + (0.69315*-1.0 + 0.69315*0.97)] / 2
        L = (0.3950955 + -0.0207945) / 2
        L = 0.1871505
        """
        expected_alpha_loss = 0.1871505
        check(out[5].numpy(), expected_alpha_loss, decimals=3)
