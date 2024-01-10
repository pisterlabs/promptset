#!/usr/bin/env python
from gym import spaces
from openai_ros.robot_envs import bebop2_env
from openai_ros.openai_ros_common import ROSLauncher
from gym.envs.registration import register
from openai_ros.task_envs.task_commons import LoadYamlFileParamsTest
import rospy
from geometry_msgs.msg import Vector3
from pathlib import Path

# The path is __init__.py of openai_ros, where we import the MovingCubeOneDiskWalkEnv directly
timestep_limit_per_episode = 1000 # Can be any Value

register(
        id='Bebop2Env-v0',
        entry_point='openai_ros.task_envs.bebop2.bebop2_task:Bebop2TaskEnv',
        max_episode_steps=timestep_limit_per_episode,
    )

class Bebop2TaskEnv(bebop2_env.Bebop2env):
    def __init__(self):

        # On load les paramètres 
        LoadYamlFileParamsTest(rospackage_name="openai_ros", rel_path_from_package_to_file="src/openai_ros/task_envs/bebop2/config", yaml_file_name="bebop2.yaml")       

        number_actions = rospy.get_param('/bebop2/n_actions')
        self.action_space = spaces.Discrete(number_actions)
        
        #parrotdrone_goto utilisait une space bos, pas nous pour l'instant.

        # Lancement de la simulation
        ROSLauncher(rospackage_name="rotors_gazebo", launch_file_name="mav_1_bebop.launch",
                    ros_ws_abspath=str(Path(__file__).parent.parent.parent.parent.parent.parent.parent))

        # Paramètres
        self.linear_forward_speed = rospy.get_param( '/bebop2/linear_forward_speed')
        self.angular_turn_speed = rospy.get_param('/bebop2/angular_turn_speed')
        self.angular_speed = rospy.get_param('/bebop2/angular_speed')

        self.init_linear_speed_vector = Vector3()

        self.init_linear_speed_vector.x = rospy.get_param( '/bebop2/init_linear_speed_vector/x')
        self.init_linear_speed_vector.y = rospy.get_param( '/bebop2/init_linear_speed_vector/y')
        self.init_linear_speed_vector.z = rospy.get_param( '/bebop2/init_linear_speed_vector/z')

        self.init_angular_turn_speed = rospy.get_param( '/bebop2/init_angular_turn_speed')


        # On charge methodes et atributs de la classe mere
        super(Bebop2TaskEnv, self).__init__()


    def _set_init_pose(self):
        """Sets the Robot in its init pose
        Appelée lorsqu'on reset la simulation
        """
        # TODO

    def _init_env_variables(self):
        """
        Inits variables needed to be initialised each time we reset at the start
        of an episode.
        :return:
        """
        self.takeoff()


    def _set_action(self, action):
        """
        Move the robot based on the action variable given
        """
        # TODO: Move robot

    def _get_obs(self):
        """
        Here we define what sensor data of our robots observations
        To know which Variables we have acces to, we need to read the
        MyRobotEnv API DOCS
        :return: observations
        """
        observations = 2
        return observations

    def _is_done(self, observations):
        """
        Decide if episode is done based on the observations
        """
        # TODO
        done = False
        return done

    def _compute_reward(self, observations, done):
        """
        Return the reward based on the observations given
        """
        reward = 0
        return reward
        
    # Internal TaskEnv Methods

