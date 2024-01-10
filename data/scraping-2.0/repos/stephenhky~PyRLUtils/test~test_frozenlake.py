
import unittest

from pyrlutils.transition import OpenAIGymDiscreteEnvironmentTransitionProbabilityFactory

class TestFrozenLake(unittest.TestCase):
    def test_factory(self):
        tranprobfactory = OpenAIGymDiscreteEnvironmentTransitionProbabilityFactory('FrozenLake-v1')
        state, actions_dict, ind_reward_fcn = tranprobfactory.generate_mdp_objects()

        assert len(state.get_all_possible_state_values()) == 16
        assert state.state_value == 0

        actions_dict[0](state)
        assert state.state_value in {0, 4}

        state.state_value = 15
        actions_dict[2](state)
        assert state.state_value == 15

        assert ind_reward_fcn(0, 0, 0) == 0.0
        assert ind_reward_fcn(14, 3, 15) == 1.0

        assert abs(tranprobfactory.get_probability(0, 0, 0) - 0.66667) < 1e-4
        assert abs(tranprobfactory.get_probability(14, 3, 15) - 0.33333) < 1e-4


if __name__ == '__main__':
    unittest.main()
