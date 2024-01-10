#!/usr/bin/env python3
import numpy as np
import time
import rospy
from pymavlink import mavutil
import os
os.environ['MAVLINK20'] = ''
from class_silver import Guidance
from std_msgs.msg import Float32,Int16,Bool

set_heading = 0 
set_depth = -20 
set_forward = 1500
gripper_data = 2
duration = 0
arm=True
# gripper_pub = rospy.Publisher('/sensor/gripper_command', Int16, queue_size=10)


def calbacktargetheading(data):
    global set_heading
    set_heading = data.data

def calbacktargetdepth(data):
    global set_depth
    set_depth = data.data

def calbacktargetforward(data):
    global set_forward
    set_forward = data.data

def callback_gripper(data):
    global gripper_data
    gripper_data = data.data

def callback_grip_duration(data):
    global duration
    duration=data.data

def callback_arming(data):
    global arm
    arm=data.data

def main ():
    global set_heading, set_depth,set_forward
    rospy.init_node('Node_Guidance', anonymous=True)
    rospy.Subscriber('target_heading',Int16,callback=calbacktargetheading)
    rospy.Subscriber('target_depth',Int16,callback=calbacktargetdepth)
    rospy.Subscriber('target_forward',Int16,callback=calbacktargetforward)
    rospy.Subscriber('target_gripper',Int16,callback=callback_gripper)
    rospy.Subscriber('target_grip_duration',Int16,callback=callback_grip_duration)
    rospy.Subscriber('target_arming',Bool,callback=callback_arming)
    # gripper_pub = rospy.Publisher('/sensor/gripper_command', Int16, queue_size=10)
    
    rospy.sleep(2)
    # rospy.Subscriber("pwm_head", Int16, pwm_head_callback)
    # master = mavutil.mavlink_connection('udpin:0.0.0.0:14550')
    master = mavutil.mavlink_connection("/dev/ttyACM0", baud=115200)  # Provide appropriate connection details
    
    print(master.wait_heartbeat())
    
    robot = Guidance(master)

    master.arducopter_arm()

    # robot.resetpwm()
    robot.setMode('MANUAL')

    # current_time = time.time()

    while not rospy.is_shutdown():
       
        robot.get_depth()
        robot.PID_depth()
        robot.control_depth()
        robot.target_pid_depth(set_depth)
        # robot.set_target_depth(-0.7)
        
        robot.PID_yaw()
        robot.control_yaw()
        robot.set_heading_target(set_heading)

        robot.control_forward(set_forward)
        robot.set_gripper(gripper_data,duration)

        if not arm :
            master.arducopter_disarm()
        
        
        
if __name__ == '__main__':
    try :
        main()
    except rospy.ROSInterruptException:
        pass

