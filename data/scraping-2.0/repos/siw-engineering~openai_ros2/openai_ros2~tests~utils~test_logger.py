import unittest
import numpy as np
from openai_ros2.utils import Logger
from openai_ros2.tasks.lobot_arm.arm_random_goal import ArmState


class TestLogger(unittest.TestCase):
    def test_serialize_deserialize(self):
        logger = Logger('table1', 'test_db.db')
        dummy_action = np.random.uniform(-1, 1, (3,))
        dummy_obs = np.random.uniform(-2.5, 2.5, (3,))
        dummy_coords = np.random.uniform(-0.2, 0.2, (3,))
        state = ArmState.Reached
        rand_num = 0.3314
        kwargs = {
            'episode_num': 10,
            'step_num': 1,
            'arm_state': state,
            'dist_to_goal': rand_num,
            'target_coords': dummy_coords,
            'current_coords': dummy_coords * 2,
            'joint_pos': dummy_obs + np.random.uniform(-0.1, 0.1, (3,)),
            'joint_pos_true': dummy_obs,
            'joint_vel': dummy_obs * 0.8 + np.random.uniform(-0.1, 0.1, (3,)),
            'joint_vel_true': dummy_obs * 0.8,
            'rew_noise': rand_num * 0.5,
            'reward': rand_num * 3,
            'normalised_reward': rand_num + np.random.rand(),
            'exp_reward': rand_num + np.random.rand() + np.random.rand(),
            'cum_unshaped_reward': rand_num * 6 + np.random.rand(),
            'cum_normalised_reward': rand_num * 8 + np.random.rand(),
            'cum_exp_reward': rand_num * 10 + np.random.rand(),
            'cum_reward': rand_num * 30,
            'cum_rew_noise': rand_num * 5,
            'action': dummy_action
        }
        for _ in range(50000):
            logger.store(**kwargs)
        loaded_data = logger.load()
        # We delete the 2 currently unstored key-value pairs such that tests still pass
        del kwargs['joint_vel_true']
        del kwargs['joint_pos_true']
        arg_vals = [v for k, v in kwargs.items()]
        loaded_vals = loaded_data[0][1:-1]
        assert len(arg_vals) == len(loaded_vals)
        for i in range(len(arg_vals)):
            loaded = loaded_vals[i]
            ori = arg_vals[i]
            if isinstance(loaded, np.ndarray) and isinstance(ori, np.ndarray):
                assert np.array_equal(loaded, ori)
            else:
                assert loaded == ori, f'failed, ori: {arg_vals[i]}, new: {loaded_vals[i]}'
        del (logger)


if __name__ == '__main__':
    unittest.main()
