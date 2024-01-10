#!/usr/bin/env python

import gym
import numpy
import time
from time import sleep
from gym import wrappers
import rospy
import rospkg
import math
from math import atan2
from tf.transformations import euler_from_quaternion
import numpy as np

# import our training environment
from openai_ros.task_envs.wamv import wamv_nav_twosets_buoys
from openai_ros import robot_gazebo_env

# messages
from robotx_gazebo.msg import UsvDrive
from nav_msgs.msg import Odometry


class Controller:

    def __init__(self):
        #copied structure from pa2
        print("Controller")

        #create the gym environment
        #rospy.init_node('wamv_nav_twosets_buoys', anonymous=True, log_level=rospy.WARN)

        # env = gym.make('WamvNavTwoSetsBuoys-v0')
        # rospy.loginfo("Gym environment done")

        # Set the logging system
        # rospack = rospkg.RosPack()
        # pkg_path = rospack.get_path('wamv_openai_ros_example')
        # outdir = pkg_path + '/training_results'
        # env = wrappers.Monitor(env, outdir, force=True)
        # rospy.loginfo("Monitor Wrapper started")


        self.usv_data = None
        self.odom_data = None
        self.current_pos = None
        self.yaw = None

        self.odom = rospy.Subscriber('/wamv/odom', Odometry, self.odom_callback)

        
        rospy.wait_for_message("/wamv/odom", Odometry)

        # ROS subscribers

        # ROS publishers
        self.cmd_drive = rospy.Publisher('/cmd_drive', UsvDrive, queue_size=1)

        self.goal = None
        self.other_goals = [(30.0, -5.0),
                            (30.0, 5.0), 
                            (-1, -5), 
                            (-1, 5)]

    def drive(self):

        if not self.goal:
            if len(self.other_goals) > 0:
                self.goal = self.other_goals.pop()
                print("Acquired a new goal: ", self.goal)
            else:
                print("We are done with all the goals!")

        v = self.goal - self.current_pos

        # are we close enough to the goal?
        DISTANCE_THRESHOLD = 2.0
        if np.linalg.norm(v) < DISTANCE_THRESHOLD:
            self.goal = None
            print("Reached the goal!")


        angle = atan2(v[1], v[0]) # from current X, Y to goal
        angle_diff = angle - self.yaw

        cmd_vel = UsvDrive()

        ANGLE_THRESHOLD = 0.1
        SPEED = 0.3
        if abs(angle_diff) < ANGLE_THRESHOLD:
            # drive straight
            left, right = 3 * SPEED, 3 * SPEED
        elif angle_diff >= 0:
            # turn left
            left, right = -SPEED, SPEED
        elif angle_diff < 0:
            # turn right
            left, right = SPEED, -SPEED

        cmd_vel.left = left
        cmd_vel.right = right

        print("Action: ", "left", left, "right", right)
            
        self.cmd_drive.publish(cmd_vel)


    def odom_callback(self, odom):
        #print("odom callback")
        #self.odom_data = data
        p = odom.pose.pose.position
        o = odom.pose.pose.orientation

        self.current_pos = np.array([p.x, p.y]) 
        (_, _, self.yaw) = euler_from_quaternion([o.x, o.y, o.z, o.w])


    def fix_angle(self, angle):
        while angle < -math.pi:
            angle += math.pi * 2
        while angle >= math.pi:
            angle -= math.pi * 2
        return angle



if __name__ == '__main__':

   # rospy.init_node('diff_drive', anonymous = True)

    rospy.init_node("diff_drive")

    
    rospy.sleep(4)  # So we don't start printing messages immediately
                    # (it can be hard to see them amidst the startup msgs)


    robot = Controller()
    r = rospy.Rate(1)
    while not rospy.is_shutdown():

        robot.drive()
        if robot.goal is None and len(robot.other_goals) == 0:
            break

        r.sleep()






    # # Create the Gym environment
    # env = gym.make('WamvNavTwoSetsBuoys-v0')
    # rospy.loginfo("Gym environment done")


    # # GOAL VALUES NEED TO BE FOUND SOMEHOW (fomr environment?)
    # goal_x = rospy.get_param("~goal_x", 10)
    # goal_y = rospy.get_param("~goal_y", 10)


    # # wait for messages where relavent
    # rospy.wait_for_message("/wamv/odom", Odometry, timeout=1.0)

    # # ROS subscribers
    # rospy.Subscriber("/wamv/odom", Odometry, calculate_velcoties)

    # # ROS publishers
    # cmd_drive = rospy.Publisher('/cmd_drive', UsvDrive, queue_size=0)

    # some relavent equations for differential kinematics of a diff drive










# import gym
# import numpy
# import time
# from gym import wrappers
# import rospy
# import rospkg
# import math

# # import our training environment
# from openai_ros.task_envs.wamv import wamv_nav_twosets_buoys
# # from gym import spaces
# # from openai_ros.robot_envs import wamv_env
# # from gym.envs.registration import register

# # messages
# from robotx_gazebo.msg import UsvDrive
# from nav_msgs.msg import Odometry

# def calculate_velcoties(msg):
#     print("here4")
#     rospy.loginfo(msg)



# def check_odom_ready():
#     odom = None
#     rospy.loginfo("Waiting for /wamv/odom to be READY...")
#     while odom is None and not rospy.is_shutdown():
#         try:
#             odom = rospy.wait_for_message("/wamv/odom", Odometry, timeout=1.0)
#             rospy.loginfo("Current /wamv/odom READY=>")

#         except:
#             rospy.logerr("Current /wamv/odom not ready yet, retrying for getting odom")
#     return odom


# if __name__ == '__main__':

#     rospy.init_node('wamv_nav_twosets_buoys', anonymous=True, log_level=rospy.WARN)

#     # # Create the Gym environment
#     # env = gym.make('WamvNavTwoSetsBuoys-v0')
#     # rospy.loginfo("Gym environment done")


#     # GOAL VALUES NEED TO BE FOUND SOMEHOW (fomr environment?)
#     # goal_x = rospy.get_param("~goal_x", 10)
#     # goal_y = rospy.get_param("~goal_y", 10)

#     #     # Get Desired Point to Get
#     # self.desired_point = Point()
#     # self.desired_point.x = rospy.get_param("/wamv/desired_point/x")
#     # self.desired_point.y = rospy.get_param("/wamv/desired_point/y")
#     # self.desired_point.z = rospy.get_param("/wamv/desired_point/z")
#     # self.desired_point_epsilon = rospy.get_param("/wamv/desired_point_epsilon")


        
#     check_odom_ready()
#     # We Start all the ROS related Subscribers and publishers
    
#     # print("here1")
#     # # wait for messages where relavent
#     # rospy.wait_for_message("/wamv/odom", Odometry, timeout=5.0)

#     # print("here2")

#     # ROS subscribers
#     rospy.Subscriber("/wamv/odom", Odometry, calculate_velcoties)


#     # print("here3")

#     # # ROS publishers
#     # cmd_drive = rospy.Publisher('/cmd_drive', UsvDrive, queue_size=1)

#     # # some relavent equations for differential kinematics of a diff drive

