from __future__ import absolute_import, division, print_function

import numpy as np
import random
import rospy
from geometry_msgs.msg import Twist
from openai_ros_envs.gazebo_connection import GazeboConnection

if __name__ == "__main__":
  gazebo = GazeboConnection(start_init_physics_parameters=True, reset_world_or_sim="WORLD")
  cmd_vel_publisher = rospy.Publisher("/mobile_base/commands/velocity", Twist, queue_size=10)
  rospy.init_node('service_test', anonymous=True, log_level=rospy.DEBUG)
  cmd_vel = Twist()
  iteration = 1
  rate = rospy.Rate(5)
  while not rospy.is_shutdown():
    print("iter: {}".format(iteration))
    for _ in range(16):
      gazebo.unpauseSim()
      rospy.logdebug("Setting cmd_vel")
      cmd_vel.linear.x = 1
      cmd_vel.angular.z = random.randrange(-2,3)
      cmd_vel_publisher.publish(cmd_vel)
      rospy.logdebug("angular: {}".format(cmd_vel.angular.z))
      rate.sleep()
      
    gazebo.pauseSim()
    gazebo.resetSim()
    rospy.logdebug("Simulation reset!!!")
    iteration += 1

    
  
  
