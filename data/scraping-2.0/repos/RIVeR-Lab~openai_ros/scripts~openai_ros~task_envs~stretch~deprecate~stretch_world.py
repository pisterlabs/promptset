#!/usr/bin/env python3

'''
LAST UPDATE: 2021.05.07

AUTHOR:     Nithish K Sanjeev Kumar
            Neset Unver Akmandor (NUA)

E-MAIL: akmandor.n@northeastern.edu

DESCRIPTION: TODO...

REFERENCES:
[1] Patel, Utsav, et al. "DWA-RL: Dynamically 
    Feasible Deep Reinforcement Learning Policy 
    for Robot Navigation among Mobile Obstacles.".

NUA TODO:
'''

import rospy
import numpy
import time
import math
import cv2
import os
import csv

from gazebo_msgs.msg import ModelStates
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Header, Bool
from squaternion import Quaternion

from gym import spaces
from gym.envs.registration import register

from gym.envs.registration import register
from geometry_msgs.msg import Vector3
from openai_ros.robot_envs import turtlebot3_env
from openai_ros.task_envs.task_commons import LoadYamlFileParamsTest
#from openai_ros.openai_ros_common import ROSLauncher


#from config import Config
#from obstacles import Obstacles
#from dwa import DWA


class TurtleBot3WorldEnv(turtlebot3_env.TurtleBot3Env):
    def __init__(self, robot_id=0, data_folder_path=""):
        """
        This Task Env is designed for having the TurtleBot3 in the turtlebot3 world
        closed room with columns.
        It will learn how to move around without crashing.
        """

        # This is the path where the simulation files, the Task and the Robot gits will be downloaded if not there
        self.robot_id = robot_id
        self.robot_namespace = "turtlebot" + str(self.robot_id)
        self.data_folder_path = data_folder_path
        self.world_name = rospy.get_param("/turtlebot3/world_name", "testing_garden_dynamic_1")

        '''
        print("==============================================================")
        print("TurtleBot3WorldEnv::__init__ -> world_name: " + str(self.world_name))
        print("==============================================================")
        '''

        ros_ws_abspath = rospy.get_param("/turtlebot3/ros_ws_abspath", None)
        assert ros_ws_abspath is not None, "You forgot to set ros_ws_abspath in your yaml file of your main RL script. Set ros_ws_abspath: \'YOUR/SIM_WS/PATH\'"
        assert os.path.exists(ros_ws_abspath), "The Simulation ROS Workspace path " + ros_ws_abspath + \
                                               " DOESNT exist, execute: mkdir -p " + ros_ws_abspath + \
                                               "/src;cd " + ros_ws_abspath + ";catkin_make"

        '''
        ROSLauncher(rospackage_name="turtlebot3_gazebo",
                    launch_file_name="start_world_wall.launch",
                    ros_ws_abspath=ros_ws_abspath)
        '''

        # Load Params from the desired Yaml file
        LoadYamlFileParamsTest(rospackage_name="openai_ros_devel",
                               rel_path_from_package_to_file="scripts/openai_ros/task_envs/turtlebot3/config",
                               yaml_file_name="turtlebot3_world.yaml")

        

        self.pedestrians_info = {}
        self.pedestrians_info["training_4robot3D1P"] = {}
        self.pedestrians_info["testing_garden_dynamic_1"] = {}
        self.pedestrians_info["testing_zigzag_static"] = {}
        self.pedestrian_pose = {}

        self.get_init_pose()
        self.get_goal_location()

        # Here we will add any init functions prior to starting the MyRobotEnv
        super(TurtleBot3WorldEnv, self).__init__(robot_namespace=self.robot_namespace, initial_pose=self.initial_pose, data_folder_path=data_folder_path)
        #super(TurtleBot3WorldEnv, self).__init__(ros_ws_abspath, robot_id=self.robot_id, initial_pose=self.initial_pose)

        # Only variable needed to be set here
        number_actions = rospy.get_param('/turtlebot3/n_actions')
        self.action_space = spaces.Discrete(number_actions)

        # We set the reward range, which is not compulsory but here we do it.
        self.reward_range = (-numpy.inf, numpy.inf)

        self.max_step_num = rospy.get_param("/turtlebot3/nsteps")


        #number_observations = rospy.get_param('/turtlebot3/n_observations')
        """
        We set the Observation space for the 6 observations
        cube_observations = [
            round(current_disk_roll_vel, 0),
            round(y_distance, 1),
            round(roll, 1),
            round(pitch, 1),
            round(y_linear_speed,1),
            round(yaw, 1),
        ]
        """

        # Actions and Observations
        self.linear_forward_speed = rospy.get_param('/turtlebot3/linear_forward_speed')
        self.linear_turn_speed = rospy.get_param('/turtlebot3/linear_turn_speed')
        self.angular_speed = rospy.get_param('/turtlebot3/angular_speed')
        self.init_linear_forward_speed = rospy.get_param('/turtlebot3/init_linear_forward_speed')
        self.init_linear_turn_speed = rospy.get_param('/turtlebot3/init_linear_turn_speed')

        self.new_ranges = rospy.get_param('/turtlebot3/new_ranges')
        self.min_range = rospy.get_param('/turtlebot3/min_range')
        self.max_laser_value = rospy.get_param('/turtlebot3/max_laser_value')
        self.min_laser_value = rospy.get_param('/turtlebot3/min_laser_value')
        self.max_linear_aceleration = rospy.get_param('/turtlebot3/max_linear_aceleration')


        # We create two arrays based on the binary values that will be assigned
        # In the discretization method.
        laser_scan = self.get_laser_scan()
        self.num_laser_readings = int(len(laser_scan.ranges)/self.new_ranges)

        '''
        print("=====================================================================================")
        print("turtlebot3_world::__init__ -> num_laser_readings: " + str(num_laser_readings))
        print("=====================================================================================")
        '''

        high = numpy.full((self.num_laser_readings), self.max_laser_value)
        low = numpy.full((self.num_laser_readings), self.min_laser_value)

        # We only use two integers
        self.observation_space = spaces.Box(low, high)

        rospy.logdebug("ACTION SPACES TYPE===>"+str(self.action_space))
        rospy.logdebug("OBSERVATION SPACES TYPE===>"+str(self.observation_space))

        # Rewards
        self.forwards_reward = rospy.get_param("/turtlebot3/forwards_reward")
        self.turn_reward = rospy.get_param("/turtlebot3/turn_reward")
        self.end_episode_points = rospy.get_param("/turtlebot3/end_episode_points")

        self.cumulated_steps = 0.0


    def _set_init_pose(self):
        """Sets the Robot in its init pose
        """
        self.move_base( self.init_linear_forward_speed,
                        self.init_linear_turn_speed,
                        epsilon=0.05,
                        update_rate=10)

        return True


    def _init_env_variables(self):
        """
        Inits variables needed to be initialised each time we reset at the start
        of an episode.
        :return:
        """
        # For Info Purposes
        self.cumulated_reward = 0.0
        # Set to false Done, because its calculated asyncronously
        self._episode_done = False
        self.step_num = 0


    def _set_action(self, action):
        """
        This set action will Set the linear and angular speed of the turtlebot2
        based on the action number given.
        :param action: The action integer that set s what movement to do next.
        """

        rospy.logdebug("Start Set Action ==>"+str(action))
        # We convert the actions to speed movements to send to the parent class CubeSingleDiskEnv
        if action == 0: #FORWARD
            linear_speed = self.linear_forward_speed
            angular_speed = 0.0
            self.last_action = "FORWARDS"
        elif action == 1: #LEFT
            linear_speed = self.linear_turn_speed
            angular_speed = self.angular_speed
            self.last_action = "TURN_LEFT"
        elif action == 2: #RIGHT
            linear_speed = self.linear_turn_speed
            angular_speed = -1*self.angular_speed
            self.last_action = "TURN_RIGHT"

        # We tell TurtleBot2 the linear and angular speed to set to execute
        self.move_base(linear_speed, angular_speed, epsilon=0.05, update_rate=10)

        self.step_num += 1

        rospy.logdebug("END Set Action ==>"+str(action))

    def _get_obs(self):
        """
        Here we define what sensor data defines our robots observations
        To know which Variables we have acces to, we need to read the
        TurtleBot2Env API DOCS
        :return:
        """

        rospy.logdebug("Start Get Observation ==>")
        # We get the laser scan data
        laser_scan = self.get_laser_scan()

        discretized_observations = self.discretize_scan_observation(    laser_scan,
                                                                        self.num_laser_readings
                                                                        )

        rospy.logdebug("Observations==>"+str(discretized_observations))
        rospy.logdebug("END Get Observation ==>")
        return discretized_observations


    def _is_done(self, observations):

        if self._episode_done:
            rospy.logerr("TurtleBot3 is Too Close to wall==>")
        else:
            rospy.logwarn("TurtleBot3 is NOT close to a wall ==>")

        if self.step_num >= self.max_step_num-1:
            rospy.logerr("turtlebot3_world::__init__ -> Max number of steps!")
            self._episode_done = True

        # Now we check if it has crashed based on the imu
        imu_data = self.get_imu()
        linear_acceleration_magnitude = self.get_vector_magnitude(imu_data.linear_acceleration)
        
        if linear_acceleration_magnitude > self.max_linear_aceleration:
            rospy.logerr("TurtleBot3 Crashed==>"+str(linear_acceleration_magnitude)+">"+str(self.max_linear_aceleration))
            self._episode_done = True
        else:
            rospy.logerr("DIDNT crash TurtleBot3 ==>"+str(linear_acceleration_magnitude)+">"+str(self.max_linear_aceleration))


        return self._episode_done

    def _compute_reward(self, observations, done):

        if not done:
            if self.last_action == "FORWARDS":
                reward = self.forwards_reward
            else:
                reward = self.turn_reward
        else:
            reward = -1*self.end_episode_points


        rospy.logdebug("reward=" + str(reward))
        self.cumulated_reward += reward
        rospy.logdebug("Cumulated_reward=" + str(self.cumulated_reward))
        self.cumulated_steps += 1
        rospy.logdebug("Cumulated_steps=" + str(self.cumulated_steps))

        return reward


    # Internal TaskEnv Methods

    def discretize_scan_observation(self,data,new_ranges):
        """
        Discards all the laser readings that are not multiple in index of new_ranges
        value.
        """
        self._episode_done = False

        discretized_ranges = []
        mod = len(data.ranges)/new_ranges

        rospy.logdebug("data=" + str(data))
        rospy.logdebug("new_ranges=" + str(new_ranges))
        rospy.logdebug("mod=" + str(mod))

        for i, item in enumerate(data.ranges):
            if (i%mod==0):
                if item == float ('Inf') or numpy.isinf(item):
                    discretized_ranges.append(self.max_laser_value)
                elif numpy.isnan(item):
                    discretized_ranges.append(self.min_laser_value)
                else:
                    discretized_ranges.append(int(item))

                if (self.min_range > item > 0):
                    rospy.logerr("done Validation >>> item=" + str(item)+"< "+str(self.min_range))
                    self._episode_done = True
                else:
                    rospy.logdebug("NOT done Validation >>> item=" + str(item)+"< "+str(self.min_range))


        return discretized_ranges


    def get_vector_magnitude(self, vector):
        """
        It calculated the magnitude of the Vector3 given.
        This is usefull for reading imu accelerations and knowing if there has been
        a crash
        :return:
        """
        contact_force_np = numpy.array((vector.x, vector.y, vector.z))
        force_magnitude = numpy.linalg.norm(contact_force_np)

        return force_magnitude

    '''
    DESCRIPTION: TODO...Gets the initial location of the robot to reset
    '''
    def get_init_pose(self):

        self.initial_pose = {}
        if (self.world_name == "training_4robot3D1P"):

            if (self.robot_id == 0):
                self.initial_pose["x_init"] = rospy.get_param('/turtlebot3/robot0_init_pos_x', 0.8866)
                self.initial_pose["y_init"] = rospy.get_param('/turtlebot3/robot0_init_pos_y', 0.24)
                robot0_init_yaw = rospy.get_param('/turtlebot3/robot0_init_yaw', 0.0)
                robot0_init_quat = Quaternion.from_euler(0, 0, robot0_init_yaw)
                self.initial_pose["x_rot_init"] = robot0_init_quat.x
                self.initial_pose["y_rot_init"] = robot0_init_quat.y
                self.initial_pose["z_rot_init"] = robot0_init_quat.z
                self.initial_pose["w_rot_init"] = robot0_init_quat.w
                self.pedestrians_info["training_4robot3D1P"][0] = [[0, "Right"],[1, "Left"]]

            elif (self.robot_id == 1):
                self.initial_pose["x_init"] = rospy.get_param('/turtlebot3/robot1_init_pos_x', 1.18)
                self.initial_pose["y_init"] = rospy.get_param('/turtlebot3/robot1_init_pos_y', 12.13)
                robot1_init_yaw = rospy.get_param('/turtlebot3/robot1_init_yaw', 0.0)
                robot1_init_quat = Quaternion.from_euler(0, 0, robot1_init_yaw)
                self.initial_pose["x_rot_init"] = robot1_init_quat.x
                self.initial_pose["y_rot_init"] = robot1_init_quat.y
                self.initial_pose["z_rot_init"] = robot1_init_quat.z
                self.initial_pose["w_rot_init"] = robot1_init_quat.w
                self.pedestrians_info["training_4robot3D1P"][1] = [[3, "Right"],[2, "Left"]]

            elif (self.robot_id == 2):
                self.initial_pose["x_init"] = rospy.get_param('/turtlebot3/robot2_init_pos_x', -10.085)
                self.initial_pose["y_init"] = rospy.get_param('/turtlebot3/robot2_init_pos_y', 12.15)
                robot2_init_yaw = rospy.get_param('robot2_init_yaw', 3.14)
                robot2_init_quat = Quaternion.from_euler(0, 0, robot2_init_yaw)
                self.initial_pose["x_rot_init"] = robot2_init_quat.x
                self.initial_pose["y_rot_init"] = robot2_init_quat.y
                self.initial_pose["z_rot_init"] = robot2_init_quat.z
                self.initial_pose["w_rot_init"] = robot2_init_quat.w
                self.pedestrians_info["training_4robot3D1P"][2] = []

            elif (self.robot_id == 3):
                self.initial_pose["x_init"] = rospy.get_param('/turtlebot3/robot3_init_pos_x', -11.0)
                self.initial_pose["y_init"] = rospy.get_param('/turtlebot3/robot3_init_pos_y', -0.03)
                robot3_init_yaw = rospy.get_param('robot3_init_yaw', 3.14)
                robot3_init_quat = Quaternion.from_euler(0, 0, robot3_init_yaw)
                self.initial_pose["x_rot_init"] = robot3_init_quat.x
                self.initial_pose["y_rot_init"] = robot3_init_quat.y
                self.initial_pose["z_rot_init"] = robot3_init_quat.z
                self.initial_pose["w_rot_init"] = robot3_init_quat.w
                self.pedestrians_info["training_4robot3D1P"][3] = [[0, "Straight"]]

        elif (self.world_name == "testing_garden_dynamic_1"):
                self.initial_pose["x_init"] = rospy.get_param('/turtlebot3/robot0_init_pos_x', 0.8866)
                self.initial_pose["y_init"] = rospy.get_param('/turtlebot3/robot0_init_pos_y', 0.24)
                robot0_init_yaw = rospy.get_param('robot0_init_yaw', 0.0)
                robot0_init_quat = Quaternion.from_euler(0, 0, robot0_init_yaw)
                self.initial_pose["x_rot_init"] = robot0_init_quat.x
                self.initial_pose["y_rot_init"] = robot0_init_quat.y
                self.initial_pose["z_rot_init"] = robot0_init_quat.z
                self.initial_pose["w_rot_init"] = robot0_init_quat.w
                self.pedestrians_info["testing_garden_dynamic_1"][0] = [[0, "Right"],[1, "Left"]]

        elif (self.world_name == "testing_zigzag_static"):
                self.initial_pose["x_init"] = rospy.get_param('/turtlebot3/robot0_init_pos_x', 5.0)
                self.initial_pose["y_init"] = rospy.get_param('/turtlebot3/robot0_init_pos_y', -8.5)
                robot0_init_yaw = rospy.get_param('/turtlebot3/robot0_init_yaw', 1.57)
                robot0_init_quat = Quaternion.from_euler(0, 0, robot0_init_yaw)
                self.initial_pose["x_rot_init"] = robot0_init_quat.x
                self.initial_pose["y_rot_init"] = robot0_init_quat.y
                self.initial_pose["z_rot_init"] = robot0_init_quat.z
                self.initial_pose["w_rot_init"] = robot0_init_quat.w
                self.pedestrians_info["testing_zigzag_static"][0] = []

        return self.initial_pose

    '''
    DESCRIPTION: TODO...Gets the goal location for each robot
    '''
    def get_goal_location(self):

        self.goal_pose = {}
        if(self.world_name == "training_4robot3D1P"):
            if (self.robot_id == 0):
                self.goal_pose["x"] = rospy.get_param('robot0_goal_pose_x', 14.81)
                self.goal_pose["y"] = rospy.get_param('robot0_goal_pose_y', 0.24)
                
            elif (self.robot_id == 1):
                self.goal_pose["x"] = rospy.get_param('robot1_goal_pose_x', 15.0)
                self.goal_pose["y"] = rospy.get_param('robot1_goal_pose_y', 12.13)
                
            elif(self.robot_id == 2):
                self.goal_pose["x"] = rospy.get_param('robot2_goal_pose_x', -24.23)
                self.goal_pose["y"] = rospy.get_param('robot2_goal_pose_y', 11.14)
                
            elif(self.robot_id == 3):
                self.goal_pose["x"] = rospy.get_param('robot3_goal_pose_x', -24.39)
                self.goal_pose["y"] = rospy.get_param('robot3_goal_pose_y', 1.021)

        elif(self.world_name == "testing_garden_dynamic_1"):
            self.goal_pose["x"] = rospy.get_param('robot0_goal_pose_x', 14.81)
            self.goal_pose["y"] = rospy.get_param('robot0_goal_pose_y', 0.24)

        elif(self.world_name == "testing_zigzag_static"):
            self.goal_pose["x"] = rospy.get_param('robot0_goal_pose_x', -11.0)
            self.goal_pose["y"] = rospy.get_param('robot0_goal_pose_y', 8.0)
