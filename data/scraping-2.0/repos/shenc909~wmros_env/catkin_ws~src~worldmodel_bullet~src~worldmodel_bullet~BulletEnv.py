#!/usr/bin/env python

import gym
# import rospy
# import roslaunch
import sys
import os
import signal
import subprocess
import time
import random
import shlex

class BulletEnv(gym.Env):
    """Superclass for all Bullet environments.
    """
    metadata = {'render.modes': ['human', 'headless','rgb_array']}

    def __init__(self):

        random_number = random.randint(10000, 15000)
        self.port = "11311"#str(random_number) #os.environ["ROS_PORT_SIM"]
        self.port = str(random_number) #os.environ["ROS_PORT_SIM"]

        os.environ["ROS_MASTER_URI"] = "http://localhost:"+self.port
        #
        self.ros_master_uri = os.environ["ROS_MASTER_URI"];

        # print("ROS_MASTER_URI=http://localhost:"+self.port + "\n")

        # self.port = os.environ.get("ROS_PORT_SIM", "11311")
        # ros_path = os.path.dirname(subprocess.check_output(["which", "roscore"]))

        # NOTE: It doesn't make sense to launch a roscore because it will be done when spawing Gazebo, which also need
        #   to be the first node in order to initialize the clock.
        # # start roscore with same python version as current script
        # self._roscore = subprocess.Popen([sys.executable, os.path.join(ros_path, b"roscore"), "-p", self.port])
        # time.sleep(1)
        # print ("Roscore launched!")
        
        # self._uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
        # self._roslaunch_parent = roslaunch.parent.ROSLaunchParent(self._uuid, roslaunch_files=[], is_core=True, show_summary=False)
        # self._roslaunch = subprocess.Popen([sys.executable, os.path.join(ros_path, b"roslaunch"), "-p", self.port, fullpath] + launch_args)
        # self._roslaunch_parent.start()

    def step(self, action):

        # Implement this method in every subclass
        # Perform a step in gazebo. E.g. move the robot
        raise NotImplementedError

    def reset(self):

        # Implemented in subclass
        raise NotImplementedError

    def _render(self, mode="human", close=False):
        pass

    def _close(self):
        pass
        # self._roslaunch_parent.shutdown()


    def _configure(self):

        # TODO
        # From OpenAI API: Provides runtime configuration to the enviroment
        # Maybe set the Real Time Factor?
        pass
    def _seed(self):

        # TODO
        # From OpenAI API: Sets the seed for this env's random number generator(s)
        pass