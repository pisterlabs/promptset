#!/usr/bin/env python


# -*- coding: utf-8 -*-

import rospy
import gym
from gym.utils import seeding
from openai_ros.gazebo_connection import GazeboConnection
from openai_ros.controllers_connection import ControllersConnection
from uav_msgs.msg import uav_pose
import numpy as np
#https://bitbucket.org/theconstructcore/theconstruct_msgs/src/master/msg/RLExperimentInfo.msg
#from openai_ros.msg import RLExperimentInfo
import os
from stable_baselines import PPO2, A2C
import socket
# if __name__ == '__main__':
#     os.environ['ROS_MASTER_URI'] = "http://10.34.29.148:11311"
#     os.environ['GAZEBO_MASTER_URI'] = "http://10.34.25.9:11351"
#     os.environ['ROS_IP'] = "10.34.29.148"
#     os.environ['ROS_HOSTNAME'] = "10.34.29.148"
#     rospy.init_node('talker2', anonymous=True)
#     rospy.spin()

# https://github.com/openai/gym/blob/master/gym/core.py
class RobotGazeboEnv(gym.Env):

    def __init__(self, robot_name_space, controllers_list, reset_controls, start_init_physics_parameters=True, reset_world_or_sim="SIMULATION",**kwargs):

        env_id = kwargs.get('env_id',1)
        self.env_id = env_id
        self.robotID = kwargs.get('robotID',1)
        self.num_robots = kwargs.get('num_robots',1)
        bashCommand ='hostname -I | cut -d' ' -f1)'
        ROSIP=socket.gethostbyname(socket.gethostname()) 
        if env_id>=0:
            #os.environ['ROS_PACKAGE_PATH']=ROS_PACKAGE_PATH
            #@HACK Setting ros and gazebo masters can be automated
            #the gazebo master is different for different environment IDs as they can run on multiple computers
            os.environ['ROS_MASTER_URI'] = "http://"+ROSIP+":1131" + str(env_id)[0]
            if env_id < 3:
                GAZEBOIP=ROSIP
                os.environ['GAZEBO_MASTER_URI'] = "http://"+GAZEBOIP+":1135" + str(env_id)[0]
            else:
                GAZEBOIP=ROSIP
                os.environ['GAZEBO_MASTER_URI'] = "http://"+GAZEBOIP+":1135" + str(env_id)[0]
            os.environ['ROS_IP'] = ROSIP
            os.environ['ROS_HOSTNAME'] = ROSIP
            rospy.init_node('firefly_env_'+str(env_id)[0]+str(self.robotID), anonymous=True, disable_signals=True)
            print("WORKER NODE " + str(env_id)[0]+str(self.robotID))


        # To reset Simulations
        rospy.logdebug("START init RobotGazeboEnv")
        self.gazebo = GazeboConnection(start_init_physics_parameters,reset_world_or_sim,self.robotID)
        self.controllers_object = ControllersConnection(namespace=robot_name_space, controllers_list=controllers_list)
        self.reset_controls = reset_controls
        self.seed()

        # Set up ROS related variables
        self.episode_num = 0
        self.cumulated_episode_reward = 0
#        self.reward_pub = rospy.Publisher('/openai/reward', RLExperimentInfo, queue_size=1)
        rospy.logdebug("END init RobotGazeboEnv")
        # self.pretrained_model = PPO2.load("/home/rtallamraju/drl_ws/logs/parallel_ppo_cartesian_att3_try6/snapshots/trained_model.pkl")
        # self.pretrained_model = PPO2.load("/home/rtallamraju/drl_ws/logs/drl_singleagent_try9/snapshots/best_model.pkl")

    def create_circle(self, radius=8):
        self.theta = [k for k in np.arange(0,6.28,0.1)]
        x = radius*np.cos(self.theta)
        y = radius*np.sin(self.theta)
        self.init_circle = [x,y]


    # Env methods
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        """
        Function executed each time step.
        Here we get the action execute it in a time step and retrieve the
        observations generated by that action.
        :param action:
        :return: obs, reward, done, info
        """

        """
        Here we should convert the action num to movement action, execute the action in the
        simulation and get the observations result of performing that action.
        """
        rospy.logdebug("START STEP OpenAIROS")


        self.gazebo.unpauseSim()
        self._set_action(action)
        obs = self._get_obs()
        done = self._is_done(obs)
        info = {}
        reward = self._compute_reward(obs, done)
        self.cumulated_episode_reward += reward

        rospy.logdebug("END STEP OpenAIROS")

        return obs, reward, done, info

    def reset(self,robotID=1):
        rospy.logwarn("Reseting RobotGazeboEnvironment")
        # if robotID == 1:
        self._reset_sim()
        self._init_env_variables()
        self._update_episode()
        obs = self._get_obs()
        rospy.logdebug("END Reseting RobotGazeboEnvironment")

        return obs

    def close(self):
        """
        Function executed when closing the environment.
        Use it for closing GUIS and other systems that need closing.
        :return:
        """
        rospy.logdebug("Closing RobotGazeboEnvironment")
        rospy.signal_shutdown("Closing RobotGazeboEnvironment")

    def _update_episode(self):
        """
        Publishes the cumulated reward of the episode and
        increases the episode number by one.
        :return:
        """
        # rospy.logwarn("PUBLISHING REWARD...")
        # self._publish_reward_topic(
        #                             self.cumulated_episode_reward,
        #                             self.episode_num
        #                             )
        rospy.logwarn("PUBLISHING REWARD...DONE="+str(self.cumulated_episode_reward)+",EP="+str(self.episode_num))

        self.episode_num += 1
        self.cumulated_episode_reward = 0


    def _publish_reward_topic(self, reward, episode_number=1):
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

    # Extension methods
    # ----------------------------

    def _reset_sim(self):
        """Resets a simulation
        """
        rospy.logdebug("RESET SIM START")
        if self.reset_controls :
            rospy.logdebug("RESET CONTROLLERS")
            self.gazebo.unpauseSim()

            self.controllers_object.reset_controllers()
            self._check_all_systems_ready()
            self._set_init_pose()
            self.gazebo.pauseSim()
            self.gazebo.resetSim(self.robotID)
            self.gazebo.unpauseSim()
            self.controllers_object.reset_controllers()
            self._check_all_systems_ready()
            self.gazebo.pauseSim()

        else:
            # if self.robotID == 1:
            rospy.logwarn("DONT RESET CONTROLLERS")
            self.gazebo.unpauseSim()
            self._check_all_systems_ready()
            self._set_init_pose()
            self.gazebo.pauseSim()
            self.gazebo.resetSim(self.robotID)
            self.gazebo.unpauseSim()
            self._check_all_systems_ready()
            # self.gazebo.pauseSim()
            # else:
                # self.gazebo.unpauseSim()

        rospy.logdebug("RESET SIM END")
        return True

    def _set_init_pose(self):
        """Sets the Robot in its init pose
        """
        self.create_circle(radius=5)
        machine_name = '/machine_'+str(self.robotID);
        self.command_topic = machine_name+"/command"
        self._cmd_vel_pub = rospy.Publisher(self.command_topic, uav_pose, queue_size=1)
        outPose = uav_pose()
        outPose.header.stamp = rospy.Time.now()
        outPose.header.frame_id="world"

        outPose.POI.x = 0
        outPose.POI.y = 0
        outPose.POI.z = 0

        r = 8
        t = np.random.choice(63,1);
        outPose.position.x = r*np.cos(self.theta[t[0]])
        outPose.position.y = r*np.sin(self.theta[t[0]])
        outPose.position.z = -r
        self._cmd_vel_pub.publish(outPose)


    def _check_all_systems_ready(self):
        """
        Checks that all the sensors, publishers and other simulation systems are
        operational.
        """
        raise NotImplementedError()

    def _get_obs(self):
        """Returns the observation.
        """
        raise NotImplementedError()

    def _init_env_variables(self):
        """Inits variables needed to be initialised each time we reset at the start
        of an episode.
        """
        raise NotImplementedError()

    def _set_action(self, action):
        """Applies the given action to the simulation.
        """
        raise NotImplementedError()

    def _is_done(self, observations):
        """Indicates whether or not the episode is done ( the robot has fallen for example).
        """
        raise NotImplementedError()

    def _compute_reward(self, observations, done):
        """Calculates the reward to give based on the observations given.
        """
        raise NotImplementedError()

    def _env_setup(self, initial_qpos):
        """Initial configuration of the environment. Can be used to configure initial state
        and extract information from the simulation.
        """
        raise NotImplementedError()