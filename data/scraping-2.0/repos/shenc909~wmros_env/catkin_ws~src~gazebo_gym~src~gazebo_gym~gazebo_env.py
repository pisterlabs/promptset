#!/usr/bin/env python

import gym
import rospy
import roslaunch
import sys
import os
import signal
import subprocess
import time
from std_srvs.srv import Empty
import random
from rosgraph_msgs.msg import Clock
import shlex
from gazebo_gym import killer

class GazeboEnv(gym.Env):
    """Superclass for all Gazebo environments.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, launchfile, launch_args=["world_name:=track1"]):
        self.last_clock_msg = Clock()

        random_number = random.randint(10000, 15000)
        self.port = "11311"#str(random_number) #os.environ["ROS_PORT_SIM"]
        self.port_gazebo = "11345"#str(random_number+1) #os.environ["ROS_PORT_SIM"]
        self.port = str(random_number) #os.environ["ROS_PORT_SIM"]
        self.port_gazebo = str(random_number+1) #os.environ["ROS_PORT_SIM"]

        os.environ["ROS_MASTER_URI"] = "http://localhost:"+self.port
        os.environ["GAZEBO_MASTER_URI"] = "http://localhost:"+self.port_gazebo
        #
        self.ros_master_uri = os.environ["ROS_MASTER_URI"];

        print("ROS_MASTER_URI=http://localhost:"+self.port + "\n")
        print("GAZEBO_MASTER_URI=http://localhost:"+self.port_gazebo + "\n")

        # self.port = os.environ.get("ROS_PORT_SIM", "11311")
        ros_path = os.path.dirname(subprocess.check_output(["which", "roscore"]))

        # NOTE: It doesn't make sense to launch a roscore because it will be done when spawing Gazebo, which also need
        #   to be the first node in order to initialize the clock.
        # # start roscore with same python version as current script
        # self._roscore = subprocess.Popen([sys.executable, os.path.join(ros_path, b"roscore"), "-p", self.port])
        # time.sleep(1)
        # print ("Roscore launched!")


        if launchfile.startswith("/"):
            fullpath = launchfile
        else:
            fullpath = os.path.join(os.path.dirname(__file__), "gym_assets", "launch", launchfile)
        if not os.path.exists(fullpath):
            raise IOError("File "+fullpath+" does not exist")
        
        self._uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
        self._cli_args = [fullpath] + launch_args
        self._roslaunch_args = launch_args
        self._roslaunch_file = [(roslaunch.rlutil.resolve_launch_arguments(self._cli_args)[0], self._roslaunch_args)]
        self._roslaunch_parent = roslaunch.parent.ROSLaunchParent(self._uuid, roslaunch_files=[], is_core=True, show_summary=False)
        # self._roslaunch = subprocess.Popen([sys.executable, os.path.join(ros_path, b"roslaunch"), "-p", self.port, fullpath] + launch_args)
        self._roslaunch_parent.start()
        # print ("Gazebo launched!")

        self.gzclient_pid = 0

        # Launch the simulation with the given launchfile name
        

        ################################################################################################################
        # r = rospy.Rate(1)
        # self.clock_sub = rospy.Subscriber('/clock', Clock, self.callback, queue_size=1000000)
        # while not rospy.is_shutdown():
        #     print("initialization: ", rospy.rostime.is_rostime_initialized())
        #     print("Wallclock: ", rospy.rostime.is_wallclock())
        #     print("Time: ", time.time())
        #     print("Rospyclock: ", rospy.rostime.get_rostime().secs)
        #     # print("/clock: ", str(self.last_clock_msg))
        #     last_ros_time_ = self.last_clock_msg
        #     print("Clock:", last_ros_time_)
        #     # print("Waiting for synch with ROS clock")
        #     # if wallclock == False:
        #     #     break
        #     r.sleep()
        ################################################################################################################

    # def callback(self, message):
    #     """
    #     Callback method for the subscriber of the clock topic
    #     :param message:
    #     :return:
    #     """
    #     # self.last_clock_msg = int(str(message.clock.secs) + str(message.clock.nsecs)) / 1e6
    #     # print("Message", message)
    #     self.last_clock_msg = message
    #     # print("Message", message)

    def step(self, action):

        # Implement this method in every subclass
        # Perform a step in gazebo. E.g. move the robot
        raise NotImplementedError

    def reset(self):

        # Implemented in subclass
        raise NotImplementedError

    def _render(self, mode="human", close=False):

        if close:
            tmp = os.popen("ps -Af").read()
            proccount = tmp.count('gzclient')
            if proccount > 0:
                if self.gzclient_pid != 0:
                    os.kill(self.gzclient_pid, signal.SIGKILL)
                    os.wait()
            return

        tmp = os.popen("ps -Af").read()
        proccount = tmp.count('gzclient')
        if proccount < 1:
            cmd = "rosrun gazebo_ros gzclient __name:=gazebo_gui /gazebo_gui/set_physics_properties:=/gazebo/set_physics_properties /gazebo_gui/get_physics_properties:=/gazebo/get_physics_properties"
            gzclient_args = shlex.split(cmd)
            subprocess.Popen(gzclient_args)
            time.sleep(5)
            self.gzclient_pid = int(subprocess.check_output(["pidof","-s","gzclient"]))
        else:
            self.gzclient_pid = 0

    def _close(self):
        # rospy.signal_shutdown('Environment Closed')
        # Kill gzclient, gzserver and roscore

        # self._roslaunch.send_signal(signal.SIGINT)
        self._roslaunch_parent.shutdown()
        # tmp = os.popen("ps -Af").read()
        # gzclient_count = tmp.count('gzclient')
        # gzserver_count = tmp.count('gzserver')
        # roscore_count = tmp.count('roscore')
        # rosmaster_count = tmp.count('rosmaster')

        # if gzclient_count > 0:
        #     os.system("killall -9 gzclient")
        # if gzserver_count > 0:
        #     os.system("killall -9 gzserver")
        
        # if rosmaster_count > 0:
        #     # os.system("killall -9 rosmaster")
        #     pass
        # if roscore_count > 0:
        #     pass
        #     # os.system("killall -9 roscore")

        # if (gzclient_count or gzserver_count or roscore_count or rosmaster_count >0):
        #     os.wait()
        
        # while os.popen("ps -Af").read().count('rosmaster') > 0:
        #     pass

        time.sleep(5)

        # rospy.signal_shutdown('env closed')

        # rospy.core._shutdown_flag = False #dont use
        
        # killer.kill() #uncomment when reverting

        # print("killer finished running")
        # while True:
        #     pass

    def _configure(self):

        # TODO
        # From OpenAI API: Provides runtime configuration to the enviroment
        # Maybe set the Real Time Factor?
        pass
    def _seed(self):

        # TODO
        # From OpenAI API: Sets the seed for this env's random number generator(s)
        pass

    def get_fullpath(self, launchfile):
        
        if launchfile.startswith("/"):
            fullpath = launchfile
        else:
            fullpath = os.path.join(os.path.dirname(__file__), "gym_assets", "launch", launchfile)
        if not os.path.exists(fullpath):
            raise IOError("File "+fullpath+" does not exist")
        
        return fullpath