import os
from datetime import datetime
from typing import Tuple, Dict

import gym
from gym.spaces import Box

import numpy

from openai_ros2.robots import LobotArmSim, LobotArmBase
from openai_ros2.utils import ut_launch, Logger
from openai_ros2.tasks import LobotArmRandomGoal, LobotArmFixedGoal

import rclpy


class LobotArmEnv(gym.Env):
    """OpenAI Gym environment for Lobot Arm, utilises continuous action space."""

    def __init__(self, robot_cls: type, task_cls: type, robot_kwargs: Dict = None, task_kwargs: Dict = None):
        if task_kwargs is None:
            task_kwargs = {}
        if robot_kwargs is None:
            robot_kwargs = {}
        ut_launch.set_network_env_vars()
        os.environ['RMW_IMPLEMENTATION'] = 'rmw_fastrtps_cpp'
        # Check if rclpy has been initialised before
        context = rclpy.get_default_context()
        if not context.ok():
            rclpy.init()
        sim_time_param = rclpy.parameter.Parameter('use_sim_time', value=True)
        self.node = rclpy.node.Node(robot_cls.__name__, parameter_overrides=[sim_time_param])
        # self.node.set_parameters([sim_time])
        self.__robot: LobotArmBase = robot_cls(self.node, **robot_kwargs)
        self.__task = task_cls(self.node, self.__robot, **task_kwargs)
        self.action_space = self.__robot.get_action_space()
        self.observation_space = self.__get_observation_space()
        # Set up ROS related variables
        self.__episode_num = 0
        self.__cumulated_episode_reward = 0
        self.__cumulated_reward_noise = 0
        self.__cumulated_norm_reward = 0
        self.__cumulated_unshaped_reward = 0
        self.__cumulated_exp_reward = 0
        self.__step_num = 0
        self.__last_done_info = None
        now = datetime.now()
        table_name = f'run_{now.strftime("%d_%m_%Y__%H_%M_%S")}'
        self.__logger = Logger(table_name)
        # self.reset()

    def step(self, action: numpy.ndarray) -> Tuple[numpy.ndarray, float, bool, dict]:
        self.__robot.set_action(action)
        robot_state: LobotArmBase.Observation = self.__robot.get_observations()
        if isinstance(self.__task, LobotArmFixedGoal):
            obs = numpy.concatenate((robot_state.position_data, robot_state.velocity_data))
        elif isinstance(self.__task, LobotArmRandomGoal):
            obs = numpy.concatenate((robot_state.position_data, robot_state.velocity_data, self.__task.target_coords))
        else:
            raise Exception(f'Task expects LobotArmFixedGoal or LobotArmRandomGoal, but received task of type {type(self.__task)}')

        done, done_info = self.__task.is_done(robot_state.noiseless_position_data, robot_state.contact_count, self.observation_space, self.__step_num)
        arm_state = done_info['arm_state']
        reward, reward_info = self.__task.compute_reward(robot_state.noiseless_position_data, arm_state)
        info: dict = {**reward_info, **done_info}
        self.__cumulated_episode_reward += reward
        self.__cumulated_reward_noise += reward_info['rew_noise']
        self.__cumulated_norm_reward += reward_info['normalised_reward']
        self.__cumulated_unshaped_reward += reward_info['normal_reward']
        self.__cumulated_exp_reward += reward_info['exp_reward']

        self.__step_num += 1
        self.__last_done_info = done_info
        log_kwargs = {
                      'episode_num': self.__episode_num,
                      'step_num': self.__step_num,
                      'arm_state': arm_state,
                      'dist_to_goal': reward_info['distance_to_goal'],
                      'target_coords': reward_info['target_coords'],
                      'current_coords': reward_info['current_coords'],
                      'joint_pos': robot_state.position_data,
                      'joint_pos_true': robot_state.noiseless_position_data,
                      'joint_vel': robot_state.velocity_data,
                      'joint_vel_true': robot_state.noiseless_velocity_data,
                      'rew_noise': reward_info['rew_noise'],
                      'reward': reward,
                      'normalised_reward': reward_info['normalised_reward'],
                      'exp_reward': reward_info['exp_reward'],
                      'cum_unshaped_reward': self.__cumulated_unshaped_reward,
                      'cum_normalised_reward': self.__cumulated_norm_reward,
                      'cum_exp_reward': self.__cumulated_exp_reward,
                      'cum_reward': self.__cumulated_episode_reward,
                      'cum_rew_noise': self.__cumulated_reward_noise,
                      'action': action
                    }
        self.__logger.store(**log_kwargs)

        # print(f"Reward for step {self.__step_num}: {reward}, \t cumulated reward: {self.__cumulated_episode_reward}")
        return obs, reward, done, info

    def reset(self):
        if self.__last_done_info is not None:
            print(f'Episode {self.__episode_num: <6}     Reward: {self.__cumulated_episode_reward:.9f}     '
                  f'Reason: {self.__last_done_info["arm_state"]:<35}      Timesteps: {self.__step_num:<4}')
        else:
            print(f'Episode {self.__episode_num: <6}     Reward: {self.__cumulated_episode_reward:.9f}     '
                  f'total timesteps: {self.__step_num:<4}')
        self.__robot.reset()
        self.__task.reset()
        self.__step_num = 0
        self.__last_done_info = None
        self.__episode_num += 1
        self.__cumulated_episode_reward = 0
        self.__cumulated_reward_noise = 0
        self.__cumulated_norm_reward = 0
        self.__cumulated_unshaped_reward = 0
        self.__cumulated_exp_reward = 0
        return numpy.zeros(self.observation_space.shape, dtype=float)

    def close(self):
        print('Closing ' + self.__class__.__name__ + ' environment.')
        self.node.destroy_node()
        rclpy.shutdown()

    def render(self, mode='human'):
        pass

    def set_state_noise(self, mu: float, sigma: float) -> None:
        self.__robot.state_noise_mu = mu
        self.__robot.state_noise_sigma = sigma

    def set_random_init_pos(self, random: bool = False) -> None:
        self.__robot.random_init_pos = random

    def __get_observation_space(self):
        joint_pos_lower_limit = numpy.array([-2.356, -1.57, -1.57])
        joint_vel_lower_limit = numpy.array([-3, -3, -3])
        target_coord_lower_limit = numpy.array([-0.16, -0.16, 0])

        joint_pos_upper_limit = numpy.array([2.356, 0.5, 1.57])
        joint_vel_upper_limit = numpy.array([3, 3, 3])
        target_coord_upper_limit = numpy.array([0.16, 0.16, 0.24])

        if isinstance(self.__task, LobotArmFixedGoal):
            lower_limits_partial = numpy.concatenate((joint_pos_lower_limit, joint_vel_lower_limit))
            upper_limits_partial = numpy.concatenate((joint_pos_upper_limit, joint_vel_upper_limit))
            return Box(lower_limits_partial, upper_limits_partial)
        elif isinstance(self.__task, LobotArmRandomGoal):
            lower_limits_full = numpy.concatenate((joint_pos_lower_limit, joint_vel_lower_limit, target_coord_lower_limit))
            upper_limits_full = numpy.concatenate((joint_pos_upper_limit, joint_vel_upper_limit, target_coord_upper_limit))
            return Box(lower_limits_full, upper_limits_full)
        else:
            raise Exception('Please set a task before getting observation space, '
                            f'or check that the task has a registered observation space. Task type: {type(self.__task)}')
