#!/usr/bin/env python
# IMPORT
import gym
import rospy
import numpy as np
import time
import random
import sys
import yaml
import math
import datetime
import csv
import rospkg
from gym import utils, spaces
from gym.utils import seeding
from gym.envs.registration import register
from scipy.spatial.transform import Rotation

# OTHER FILES
# import util_env as U
# import math_util as UMath
from joint_array_publisher import JointArrayPub
import logger
# MESSAGES/SERVICES
from std_msgs.msg import String
from std_msgs.msg import Bool
from sensor_msgs.msg import JointState

from sensor_msgs.msg import Image

from geometry_msgs.msg import Point, Quaternion, Vector3

from geometry_msgs.msg import Pose
from geometry_msgs.msg import Point, PoseStamped
from openai_ros.msg import RLExperimentInfo
from moveit_msgs.msg import MoveGroupActionFeedback

register(
    id='AuboReach-v4',
    entry_point='aubo_reach7_env:PickbotEnv',
    max_episode_steps=50,
)


# DEFINE ENVIRONMENT CLASS
class PickbotEnv(gym.GoalEnv):

    def __init__(self, joint_increment=None, sim_time_factor=0.005, random_object=False, random_position=False,
                 use_object_type=False, populate_object=False, env_object_type='free_shapes'):

        rospy.init_node('env_node', anonymous=True)
        self._seed()

        # Assign Parameters
        self._joint_increment = joint_increment  # joint_increment in rad
        self._random_object = random_object
        self._random_position = random_position
        self._use_object_type = use_object_type
        self._populate_object = populate_object

        # Assign MsgTypes
        self.joints_state = JointState()
        self.current_pose_moveit = PoseStamped()

        self.movement_complete = Bool()
        self.movement_complete.data = False
        self.moveit_action_feedback = MoveGroupActionFeedback()
        self.feedback_list = []

        self.publisher_to_moveit_object = JointArrayPub()

        rospy.Subscriber("/joint_states", JointState, self.joints_state_callback)

        rospy.Subscriber("/pickbot/movement_complete", Bool, self.movement_complete_callback)
        rospy.Subscriber("/move_group/feedback", MoveGroupActionFeedback, self.move_group_action_feedback_callback,
                         queue_size=4)

        self.new_action = [0, 0, 0, 0]

        self.action_space = spaces.Box(-1.7, 1.7, shape=(4,), dtype="float32")

        self.goal = np.array([-0.503, 0.605, -1.676, 1.391])

        self.check_joint_states()
        obs = self._get_obs()
        self.observation_space = spaces.Dict(
            dict(
                desired_goal=spaces.Box(
                    -np.inf, np.inf, shape=obs["achieved_goal"].shape, dtype="float32"
                ),
                achieved_goal=spaces.Box(
                    -np.inf, np.inf, shape=obs["achieved_goal"].shape, dtype="float32"
                ),
                observation=spaces.Box(
                    -np.inf, np.inf, shape=obs["observation"].shape, dtype="float32"
                ),
            )
        )
        self.counter = 1
        # self.observation_space = spaces.Box(low, high)
        self.reward_range = (-np.inf, np.inf)
        # self.reward_range = (0, 0.9)
        print("------------------start seed-------------------------")
        print("-------------------exit seed-----------------")
        self.done_reward = 0

        # set up everything to publish the Episode Number and Episode Reward on a rostopic
        self.episode_num = 0
        self.accumulated_episode_reward = 0
        self.episode_steps = 0
        self.reward_pub = rospy.Publisher('/openai/reward', RLExperimentInfo, queue_size=1)
        self.reward_list = []
        self.episode_list = []
        self.step_list = []
        self.csv_name = logger.get_dir() + '/result_log'
        print("CSV NAME")
        print(self.csv_name)
        self.csv_success_exp = "success_exp" + datetime.datetime.now().strftime('%Y-%m-%d_%Hh%Mmin') + ".csv"

        # self.reset()

    def goal_distance(self, goal_a, goal_b):
        assert goal_a.shape == goal_b.shape
        return np.linalg.norm(goal_a - goal_b, axis=-1)

    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.
        d = self.goal_distance(achieved_goal, goal)
        # print("reward distance: {}".format(-d))
        return -(d > 0.1).astype(np.float32)

    # Callback Functions for Subscribers to make topic values available each time the class is initialized
    def joints_state_callback(self, msg):
        self.joints_state = msg

    def movement_complete_callback(self, msg):
        self.movement_complete = msg

    def move_group_action_feedback_callback(self, msg):
        self.moveit_action_feedback = msg

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):

        self._check_all_systems_ready()

        observation = self._get_obs()

        self._update_episode()
        return observation

    def get_distance_gripper_to_object(self, gripper_joints, height=None):

        Object = self.goal
        Gripper = np.asarray(gripper_joints)

        distance = np.linalg.norm(Object - Gripper)

        return distance

    def step(self, action):

        print("############################")

        self.movement_complete.data = False

        self.publisher_to_moveit_object.pub_joints_to_moveit(action)
        rospy.wait_for_message("/pickbot/movement_complete", Bool)
        while True:
            if self.movement_complete.data == True:
                break
        new_action = np.append(action, [0.0, 0.0])

        current_joint = np.asarray(
            [self.joints_state.position[0], self.joints_state.position[1], self.joints_state.position[2],
             self.joints_state.position[3]])
        start_ros_time = rospy.Time.now()
        print(action)
        obs = self._get_obs()
        while True:
            elapsed_time = rospy.Time.now() - start_ros_time

            if np.linalg.norm(action - obs["achieved_goal"], axis=-1) < 0.01:
                break
            elif elapsed_time > rospy.Duration(10):
                break

        print('action: {}'.format(action))
        print('achieved: {}'.format(obs['achieved_goal']))

        done = False

        info = {
            "is_success": self._is_success(obs["achieved_goal"], self.goal),
        }
        reward = self.compute_reward(obs["achieved_goal"], self.goal, info)

        self.episode_steps += 1

        self.accumulated_episode_reward += reward
        # print("======================the reward==========================")
        print(reward)
        # print('==========================================================')
        # row_list = [reward, self.counter]
        # with open('rewards.csv', 'a', encoding='UTF8', newline='') as f:
        #     writer = csv.writer(f)
        #
        #     # write the header
        #     writer.writerow(row_list)
        # self.counter = self.counter + 1
        return obs, reward, done, info

    def _check_all_systems_ready(self):
        """
        Checks that all subscribers for sensortopics are working
        1) /joint_states
        """
        self.check_joint_states()
        rospy.logdebug("ALL SYSTEMS READY")

    def check_joint_states(self):
        joint_states_msg = None
        while joint_states_msg is None and not rospy.is_shutdown():
            try:
                joint_states_msg = rospy.wait_for_message("/joint_states", JointState, timeout=0.1)
                self.joints_state = joint_states_msg
                rospy.logdebug("Current joint_states READY")
            except Exception as e:
                rospy.logdebug("Current joint_states not ready yet, retrying==>" + str(e))
                print("EXCEPTION: Joint States not ready yet, retrying.")

    def get_action_to_position(self, action, last_position):
        """
        takes the last position and adds the increments for each joint
        returns the new position
        """
        # action_position = np.asarray(last_position) + action
        action_position = last_position + action
        # clip action that is going to be published to -2.9 and 2.9 just to make sure to avoid loosing controll of controllers
        x = np.clip(action_position, -(math.pi - 0.05), math.pi - 0.05)

        return x.tolist()

    def _get_obs(self):
        """
        Returns the state of the robot needed for Algorithm to learn
        The state will be defined by a List (later converted to numpy array) of the:
        1)          Distance from desired point in meters
        2-7)        States of the 6 joints in radiants
        8,9)        Force in contact sensor in Newtons
        10,11,12)   x, y, z Position of object?
        MISSING
        10)     RGBD image


        """
        joint_states = self.joints_state
        # print("==================================================================================")
        shoulder_joint_state = joint_states.position[0]
        foreArm_joint_state = joint_states.position[1]
        upperArm_joint_state = joint_states.position[2]
        wrist_joint_state = joint_states.position[3]

        for joint in joint_states.position:
            if joint > 2 * math.pi or joint < -2 * math.pi:
                print(joint_states.name)
                print(np.around(joint_states.position, decimals=3))
                sys.exit("Joint exceeds limit")

        self.curr_joint = np.array([shoulder_joint_state, foreArm_joint_state, upperArm_joint_state, wrist_joint_state])
        object = self.goal
        # rel_pos = self.get_distance_gripper_to_object(self.joints_state.position)

        achieved_goal = self.curr_joint.copy()
        rel_pos = achieved_goal - object
        # relative_pose = np.asarray(rel_pos)
        obs = np.concatenate([achieved_goal, rel_pos])
        # Stack all information into Observations List

        return {
            "observation": obs.copy(),
            "achieved_goal": achieved_goal.copy(),
            "desired_goal": self.goal.copy(),
        }

    def _is_success(self, achieved_goal, desired_goal):
        d = self.goal_distance(achieved_goal, desired_goal)
        return (d < 0.1).astype(np.float32)

    def _update_episode(self):
        """
        Publishes the accumulated reward of the episode and
        increases the episode number by one.
        :return:
        """
        if self.episode_num > 0:
            self._publish_reward_topic(
                self.accumulated_episode_reward,
                self.episode_steps,
                self.episode_num
            )

        self.episode_num += 1
        self.accumulated_episode_reward = 0
        self.episode_steps = 0

    def _publish_reward_topic(self, reward, steps, episode_number=1):
        """
        This function publishes the given reward in the reward topic for
        easy access from ROS infrastructure.
        :param reward:
        :param episode_number:
        :return:
        """
        reward_msg = RLExperimentInfo()
        reward_msg.episode_number = episode_number
        reward_msg.episode_reward = reward
        self.reward_pub.publish(reward_msg)
        self.reward_list.append(reward)
        self.episode_list.append(episode_number)
        self.step_list.append(steps)
        list = str(reward) + ";" + str(episode_number) + ";" + str(steps) + "\n"

        with open(self.csv_name + '.csv', 'a') as csv:
            csv.write(str(list))