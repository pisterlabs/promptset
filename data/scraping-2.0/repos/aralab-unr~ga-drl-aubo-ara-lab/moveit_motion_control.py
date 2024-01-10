#!/usr/bin/env python
# python2.7
import os
import cv2
import sys
import time
import math
import rospy
import copy
import tf
import numpy as np
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
# from robotiq_c_model_control.msg import _CModel_robot_output as outputMsg

from cv_bridge import CvBridge, CvBridgeError
from scipy.misc import imsave

# MESSAGES/SERVICES
from std_msgs.msg import String
from std_msgs.msg import Float64
from std_msgs.msg import Bool
from sensor_msgs.msg import JointState, Image
from gazebo_msgs.msg import ContactsState
from sensor_msgs.msg import Image
from gazebo_msgs.srv import GetModelState
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.srv import GetLinkState
from geometry_msgs.msg import Point, Quaternion, Vector3
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Point
from openai_ros.msg import RLExperimentInfo

##___GLOBAL VARIABLES___###


##___INITIALIZATION___###
moveit_commander.roscpp_initialize(sys.argv)  # initialize the moveit commander
rospy.init_node('move_group_python_interface', anonymous=True)  # initialize the node
robot = moveit_commander.RobotCommander()  # define the robot
scene = moveit_commander.PlanningSceneInterface()  # define the scene
group = moveit_commander.MoveGroupCommander(
    "manipulator_i5")  # define the planning group (from the moveit packet 'manipulator' planning group)
display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path',
                                               moveit_msgs.msg.DisplayTrajectory)  # publisher that publishes a plan to the topic: '/move_group/display_planned_path'
# gripper_publisher = rospy.Publisher('CModelRobotOutput', outputMsg.CModel_robot_output)
tf_listener = tf.TransformListener()
tf_broadcaster = tf.TransformBroadcaster()


def joint_position_sub():
    rospy.Subscriber('/pickbot/target_joint_positions', JointState, joint_callback)
    rospy.Subscriber('/pickbot/relative_joint_positions', JointState, relative_joint_callback)


def joint_callback(data):
    pos = data.position
    pub = rospy.Publisher('/pickbot/movement_complete', Bool, queue_size=10)
    complete_msg = Bool()
    complete_msg.data = False
    check_publishers_connection(pub)
    pub.publish(complete_msg)

    # assign_joint_value(pos[2], pos[1], pos[0], pos[3], pos[4], pos[5])
    assign_joint_value(pos[0], pos[1], pos[2], pos[3])

    # check_publishers_connection(pub)
    complete_msg.data = True
    pub.publish(complete_msg)


def relative_joint_callback(data):
    pos = data.position
    pub = rospy.Publisher('/pickbot/movement_complete', Bool, queue_size=10)
    complete_msg = Bool()
    complete_msg.data = False
    check_publishers_connection(pub)
    pub.publish(complete_msg)

    relative_joint_value(pos[3], pos[2], pos[1], pos[0])

    # check_publishers_connection(pub)
    complete_msg.data = True
    pub.publish(complete_msg)


def check_publishers_connection(pub):
    """
    Checks that all the publishers are working
    :return:
    """
    rate = rospy.Rate(100)  # 10hz
    while (pub.get_num_connections() == 0):
        rospy.logdebug("No subscribers to _joint1_pub yet so we wait and try again")
        try:
            rate.sleep()
        except rospy.ROSInterruptException:
            # This is to avoid error when world is rested, time when backwards.
            pass
    rospy.logdebug("joint_pub Publisher Connected")


def randomly_spawn_object():
    """
    spawn the object unit_box_0 in a random position in the shelf
    """
    try:
        spawn_box = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        box = ModelState()
        box.model_name = "unit_box_0"
        box.pose.position.x = np.random.uniform(low=-0.35, high=0.3, size=None)
        box.pose.position.y = np.random.uniform(low=0.7, high=0.9, size=None)
        box.pose.position.z = 1.05
        spawn_box(box)
    except rospy.ServiceException as e:
        rospy.loginfo("Set Model State service call failed:  {0}".format(e))


###___REGRASP FUNCTION___###
## Regrasp thin object by simultaneously tiliting end-effector and widening grip (unit: mm)
def regrasp(theta, length, phi_target, axis, direction, tilt_axis,
            tilt_direction):  # Assumption is that initial conditions are phi = 0 and opposite = length

    resol = 1  # set resolution of incremental movements with respect to phi (unit: degrees)
    rate_hz = 10  # set speed of regrasp by setting update frequency (hz)
    phi_current = 0.0
    i = 1
    while phi_current < phi_target:

        opposite = length * math.sin(math.radians(90 - phi_current))

        center_of_rotation = get_instantaneous_center(opposite, rate_hz)

        width = opposite / math.tan(math.radians(90 - phi_current + 1))
        position = int((width - 146.17) / (-0.6584))  # Gripper position from a range of (0-255)
        phi_current = phi_current + resol
        i += 1
        set_gripper_position(position)  # increment gripper width
        if axis is 'x':
            TurnArcAboutAxis('x', center_of_rotation[1], center_of_rotation[2], resol, direction, 'yes', tilt_axis,
                             tilt_direction)
        if axis is 'y':
            TurnArcAboutAxis('y', center_of_rotation[2], center_of_rotation[0], resol, direction, 'yes', tilt_axis,
                             tilt_direction)
        if axis is 'z':
            TurnArcAboutAxis('z', center_of_rotation[0], center_of_rotation[1], resol, direction, 'yes', tilt_axis,
                             tilt_direction)
            # print 'Position: ', position, ' CoR: ', center_of_rotation #' phi_current: ', phi_current, ' width: ', width, ' opposite: ', opposite #debug


## Get instantaneous center of rotation for regrasp() function
def get_instantaneous_center(opposite, rate_hz):
    rate = rospy.Rate(rate_hz)
    displacement = 0.277 - (opposite / 2) / 1000

    tf_listener.waitForTransform('/base_link', '/ee_link', rospy.Time(), rospy.Duration(4.0))
    (trans1, rot1) = tf_listener.lookupTransform('/base_link', '/ee_link',
                                                 rospy.Time(0))  # listen to transform between base_link2ee_link
    base2eelink_matrix = tf_listener.fromTranslationRotation(trans1,
                                                             rot1)  # change base2eelink from transform to matrix
    eelink2eetip_matrix = tf_listener.fromTranslationRotation((displacement, 0.0, 0.0), (
    0.0, 0.0, 0.0, 1.0))  # change eelink2eetip from transform to matrix
    base2eetip_matrix = np.matmul(base2eelink_matrix,
                                  eelink2eetip_matrix)  # combine transformation: base2eetip = base2eelink x eelink2eetip
    scale, shear, rpy_angles, translation_vector, perspective = tf.transformations.decompose_matrix(
        base2eetip_matrix)  # change base2eetip from matrix to transform
    quaternion = tf.transformations.quaternion_from_euler(rpy_angles[0], rpy_angles[1], rpy_angles[2])
    rate.sleep()
    return translation_vector
    # print translation_vector, quaternion #debug
    # print base2eetip_matrix #debug
    # print base2eelink_matrix #debug


###___TURN ARC FUNCTION___###
## Turns about a reference center point in path mode or tilt mode
## User specifies axis:['x'/'y'/'z'], Center of Circle: [y,z / z,x / x,y], Arc turn angle: [degrees], Direction: [1/-1], Tilt Mode: ['yes'/'no'], End_effector tilt axis: ['x'/'y'/'z'], Tilt direction: [1/-1]
def TurnArcAboutAxis(axis, CenterOfCircle_1, CenterOfCircle_2, angle_degree, direction, tilt, tilt_axis,
                     tilt_direction):
    pose_target = group.get_current_pose().pose  # create a pose variable. The parameters can be seen from "$ rosmsg show Pose"
    waypoints = []
    waypoints.append(pose_target)
    resolution = 360  # Calculation of resolution by (180/resolution) degrees
    # define the axis of rotation
    if axis is 'x':
        position_1 = pose_target.position.y
        position_2 = pose_target.position.z
    if axis is 'y':
        position_1 = pose_target.position.z
        position_2 = pose_target.position.x
    if axis is 'z':
        position_1 = pose_target.position.x
        position_2 = pose_target.position.y

    circle_radius = ((position_1 - CenterOfCircle_1) ** 2 + (
                position_2 - CenterOfCircle_2) ** 2) ** 0.5  # Pyth. Theorem to find radius

    # calculate the global angle with respect to 0 degrees based on which quadrant the end_effector is in
    if position_1 > CenterOfCircle_1 and position_2 > CenterOfCircle_2:
        absolute_angle = math.asin(math.fabs(position_2 - CenterOfCircle_2) / circle_radius)
    if position_1 < CenterOfCircle_1 and position_2 > CenterOfCircle_2:
        absolute_angle = math.pi - math.asin(math.fabs(position_2 - CenterOfCircle_2) / circle_radius)
    if position_1 < CenterOfCircle_1 and position_2 < CenterOfCircle_2:
        absolute_angle = math.pi + math.asin(math.fabs(position_2 - CenterOfCircle_2) / circle_radius)
    if position_1 > CenterOfCircle_1 and position_2 < CenterOfCircle_2:
        absolute_angle = 2.0 * math.pi - math.asin(math.fabs(position_2 - CenterOfCircle_2) / circle_radius)

    theta = 0  # counter that increases the angle
    while theta < angle_degree / 180.0 * math.pi:
        if axis is 'x':
            pose_target.position.y = circle_radius * math.cos(
                theta * direction + absolute_angle) + CenterOfCircle_1  # equation of circle from polar to cartesian x = r*cos(theta)+dx
            pose_target.position.z = circle_radius * math.sin(
                theta * direction + absolute_angle) + CenterOfCircle_2  # equation of cirlce from polar to cartesian y = r*sin(theta)+dy
        if axis is 'y':
            pose_target.position.z = circle_radius * math.cos(theta * direction + absolute_angle) + CenterOfCircle_1
            pose_target.position.x = circle_radius * math.sin(theta * direction + absolute_angle) + CenterOfCircle_2
        if axis is 'z':
            pose_target.position.x = circle_radius * math.cos(theta * direction + absolute_angle) + CenterOfCircle_1
            pose_target.position.y = circle_radius * math.sin(theta * direction + absolute_angle) + CenterOfCircle_2

        ## Maintain orientation with respect to turning axis
        if tilt is 'yes':
            pose_target = TiltAboutAxis(pose_target, resolution, tilt_axis, tilt_direction)

        waypoints.append(copy.deepcopy(pose_target))
        theta += math.pi / resolution  # increment counter, defines the number of waypoints
    del waypoints[:2]
    plan_execute_waypoints(waypoints)


def TiltAboutAxis(pose_target, resolution, tilt_axis, tilt_direction):
    quaternion = (
        pose_target.orientation.x,
        pose_target.orientation.y,
        pose_target.orientation.z,
        pose_target.orientation.w)

    # euler = quaternion_to_euler(quaternion[0], quaternion[1], quaternion[2], quaternion[3])
    euler = tf.transformations.euler_from_quaternion(quaternion)  # convert quaternion to euler
    roll = euler[0]
    pitch = euler[1]
    yaw = euler[2]
    # increment the orientation angle
    if tilt_axis is 'x':
        roll += tilt_direction * math.pi / resolution
    if tilt_axis is 'y':
        pitch += tilt_direction * math.pi / resolution
    if tilt_axis is 'z':
        yaw += tilt_direction * math.pi / resolution
    quaternion = tf.transformations.quaternion_from_euler(roll, pitch, yaw)  # convert euler to quaternion
    # store values to pose_target
    pose_target.orientation.x = quaternion[0]
    pose_target.orientation.y = quaternion[1]
    pose_target.orientation.z = quaternion[2]
    pose_target.orientation.w = quaternion[3]
    return pose_target


###___JOINT VALUE MANIPULATION___###
## Manipulate by assigning joint values
def assign_joint_value(joint_0, joint_1, joint_2, joint_3):
    # group.set_max_velocity_scaling_factor(0.1)
    group_variable_values = group.get_current_joint_values()  # create variable that stores joint values
    # print("group: {}".format(np.round(group_variable_values, decimals=3)))

    # Assign values to joints
    group_variable_values[0] = joint_0
    group_variable_values[1] = joint_1
    group_variable_values[2] = joint_2
    group_variable_values[3] = joint_3

    group.set_joint_value_target(group_variable_values)  # set target joint values for 'manipulator' group

    plan = group.plan()  # call plan function to plan the path (visualize on rviz)
    group.go(wait=True)  # execute plan on real/simulation (gazebo) robot
    group.stop()
    rospy.sleep(0.1)


###___POSE TARGET MANIPULATION___###
## Manipulate by assigning pose target
def assign_pose_target(pos_x, pos_y, pos_z, orient_x, orient_y, orient_z, orient_w):
    group.set_max_velocity_scaling_factor(0.1)
    pose_target = group.get_current_pose()  # create a pose variable. The parameters can be seen from "$ rosmsg show Pose"

    # Assign values
    if pos_x is 'nil':
        pass
    else:
        pose_target.pose.position.x = pos_x
    if pos_y is 'nil':
        pass
    else:
        pose_target.pose.position.y = pos_y
    if pos_z is 'nil':
        pass
    else:
        pose_target.pose.position.z = pos_z
    if orient_x is 'nil':
        pass
    else:
        pose_target.pose.orientation.x = orient_x
    if orient_y is 'nil':
        pass
    else:
        pose_target.pose.orientation.y = orient_y
    if orient_z is 'nil':
        pass
    else:
        pose_target.pose.orientation.z = orient_z
    if orient_w is 'nil':
        pass
    else:
        pose_target.pose.orientation.w = orient_w

    group.set_pose_target(pose_target)  # set pose_target as the goal pose of 'manipulator' group

    plan2 = group.plan()  # call plan function to plan the path
    group.go(wait=True)  # execute plan on real/simulation robot
    rospy.sleep(2)  # sleep 5 seconds


###___RELATIVE JOINT VALUE MANIPULATION___###
## Manipulate by assigning relative joint values w.r.t. current joint values of robot
def relative_joint_value(joint_0, joint_1, joint_2, joint_3):
    group.set_max_velocity_scaling_factor(0.1)
    group_variable_values = group.get_current_joint_values()  # create variable that stores joint values

    # Assign values to joints
    group_variable_values[0] += joint_0
    group_variable_values[1] += joint_1
    group_variable_values[2] += joint_2
    group_variable_values[3] += joint_3

    group.set_joint_value_target(group_variable_values)  # set target joint values for 'manipulator' group

    plan1 = group.plan()  # call plan function to plan the path (visualize on rviz)
    group.go(wait=True)  # execute plan on real/simulation (gazebo) robot
    # rospy.sleep(2) #sleep 2 seconds


###___RELATIVE POSE TARGET MANIPULATION___###
## Manipulate by moving gripper linearly with respect to world frame
def relative_pose_target(axis_world, distance):
    group.set_max_velocity_scaling_factor(0.1)
    pose_target = group.get_current_pose()  # create a pose variable. The parameters can be seen from "$ rosmsg show Pose"
    if axis_world is 'x':
        pose_target.pose.position.x += distance
    if axis_world is 'y':
        pose_target.pose.position.y += distance
    if axis_world is 'z':
        pose_target.pose.position.z += distance
    group.set_pose_target(pose_target)  # set pose_target as the goal pose of 'manipulator' group

    plan2 = group.plan()  # call plan function to plan the path
    group.go(wait=True)  # execute plan on real/simulation robot
    rospy.sleep(2)  # sleep 5 seconds


def plan_execute_waypoints(waypoints):
    (plan3, fraction) = group.compute_cartesian_path(waypoints, 0.01,
                                                     0)  # parameters(waypoints, resolution_1cm, jump_threshold)
    plan = group.retime_trajectory(robot.get_current_state(), plan3, 0.1)  # parameter that changes velocity
    group.execute(plan)


###___STATUS ROBOT___###
def manipulator_status():
    # You can get a list with all the groups of the robot like this:
    print("Robot Groups:")
    print(robot.get_group_names())

    # You can get the current values of the joints like this:
    print("Current Joint Values:")
    print(group.get_current_joint_values())

    # You can also get the current Pose of the end effector of the robot like this:
    print("Current Pose:")
    print(group.get_current_pose())

    # Finally you can check the general status of the robot like this:
    print("Robot State:")
    print(robot.get_current_state())


###___Initiate node; subscribe to topic; call callback function___###
def manipulator_arm_control():
    ###___LIST OF FUNCTIONS___###
    ##      assign_pose_target(pos_x, pos_y, pos_z, orient_x, orient_y, orient_z, orient_w)
    ##      assign_joint_value(joint_0, joint_1, joint_2, joint_3, joint_4, joint_5)
    ##      relative_pose_target(axis_world, distance)
    ##      relative_joint_value(joint_0, joint_1, joint_2, joint_3, joint_4, joint_5)
    ##      TurnArcAboutAxis(axis, CenterOfCircle_1, CenterOfCircle_2, angle_degree, direction, tilt, tilt_axis, tilt_direction)
    ##      regrasp(theta, length, phi_target, axis, direction, tilt_axis, tilt_direction)

    ###___TEMP___###

    ###___MOTION PLAN TO SET ROBOT TO REAL ENVIRONMNET for GAZEBO___###
    #    relative_joint_value(0, -math.pi/2, 0, 0, 0, 0)
    #    relative_joint_value(0, 0, -3*math.pi/4, 0, 0, 0)
    #    relative_joint_value(0, 0, 0, -3*math.pi/4, 0, 0)
    #    relative_joint_value(0, 0, 0, 0, -math.pi/2, 0)
    # assign_pose_target(-0.52, 0.1166, 0.22434, 0.0, 0.707, -0.707, 0.0) ## REAL ROBOT ENVIRONMENT

    ###___REGRASP DEMO___###
    # assign_pose_target(-0.52, 0.1166, 0.22434, 0.0, 0.707, -0.707, 0.0)
    # TurnArcAboutAxis('y', 0.22434, -0.79, 60, -1, 'yes', 'y', 1)
    # regrasp(60.0, 50.0, 20.0, 'y', -1, 'y', 1)

    ###___TURNARC DEMO___###
    # assign_pose_target(-0.52, 0.1166, 0.22434, 0.0, 0.707, -0.707, 0.0) ## REAL ROBOT ENVIRONMENT
    # TurnArcAboutAxis('y', 0.22434, -0.79, 40, -1, 'yes', 'y', 1)
    # TurnArcAboutAxis('y', 0.22434, -0.79, 40, 1, 'yes', 'y', -1)
    # TurnArcAboutAxis('y', 0.22434, -0.79, 70, -1, 'yes', 'y', 1)
    # TurnArcAboutAxis('y', 0.22434, -0.79, 70, 1, 'yes', 'y', -1)
    # TurnArcAboutAxis('y', 0.22434, -0.79, 10, 1, 'yes', 'y', -1)
    # TurnArcAboutAxis('y', 0.22434, -0.79, 10, -1, 'yes', 'y', 1)

    print("1. Moving to position 1")
    assign_pose_target(0.4, 0.5, 0.6, 0.2, 0.0, 0.0, 0.0)
    print("Current position 1: {},{},{}".format(group.get_current_pose().pose.position.x,
                                                group.get_current_pose().pose.position.y,
                                                group.get_current_pose().pose.position.z))
    '''
    print("--------Taking a picture")
    photoShooter = PhotoShooter()
    photoShooter.main()
    '''

    print("2. Moving to position 2")
    assign_pose_target(0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0)
    print("Current position 2: {},{},{}".format(group.get_current_pose().pose.position.x,
                                                group.get_current_pose().pose.position.y,
                                                group.get_current_pose().pose.position.z))
    '''
    print("--------Taking a picture")
    photoShooter.main()
    '''

    print("3. Moving to position 3")
    assign_pose_target(0.01, 1.0, 0.8, 0.0, 0.0, 0.0, 0.0)
    print("Current position 3: {},{},{}".format(group.get_current_pose().pose.position.x,
                                                group.get_current_pose().pose.position.y,
                                                group.get_current_pose().pose.position.z))
    '''
    print("--------Taking a picture")
    photoShooter.main()
    '''

    print("Randomly spawning object now")
    randomly_spawn_object()

    print("4. Moving to position 4")
    assign_pose_target(-0.1, 0.9, 0.21, 0.0, 0.0, 0.0, 0.0)
    print("Current position 4: {},{},{}".format(group.get_current_pose().pose.position.x,
                                                group.get_current_pose().pose.position.y,
                                                group.get_current_pose().pose.position.z))

    print("Assigning joint values")
    relative_joint_value(0, 0, 0, 0, 0, math.pi / 2)

    print("Planning ended.")

    rospy.spin()


def get_distance_gripper_to_object():
    """
    Get the Position of the endeffektor and the object via rosservice call /gazebo/get_model_state and /gazebo/get_link_state
    Calculate distance between them
    In this case
    Object:     unite_box_0 link
    Gripper:    vacuum_gripper_link ground_plane
    """

    try:
        model_coordinates = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        blockName = "unit_box_0"
        relative_entity_name = "link"
        object_resp_coordinates = model_coordinates(blockName, relative_entity_name)
        Object = np.array((object_resp_coordinates.pose.position.x, object_resp_coordinates.pose.position.y,
                           object_resp_coordinates.pose.position.z))

        print("Object: {}".format(Object))

    except rospy.ServiceException as e:
        rospy.loginfo("Get Model State service call failed:  {0}".format(e))
        print("Exception get model state")

    try:
        model_coordinates = rospy.ServiceProxy('/gazebo/get_link_state', GetLinkState)
        LinkName = "vacuum_gripper_link"
        ReferenceFrame = "ground_plane"
        resp_coordinates_gripper = model_coordinates(LinkName, ReferenceFrame)
        Gripper = np.array((resp_coordinates_gripper.link_state.pose.position.x,
                            resp_coordinates_gripper.link_state.pose.position.y,
                            resp_coordinates_gripper.link_state.pose.position.z))

        print("Gripper position: {},{},{}".format(resp_coordinates_gripper.link_state.pose.position.x,
                                                  resp_coordinates_gripper.link_state.pose.position.y,
                                                  resp_coordinates_gripper.link_state.pose.position.z))

    except rospy.ServiceException as e:
        rospy.loginfo("Get Link State service call failed:  {0}".format(e))
        print("Exception get Gripper position")
    distance = np.linalg.norm(Object - Gripper)

    return distance, Object


'''
moving = True
new_image = False
img_idx = 0
depth_img_idx = 0
def save_image(cv_image, index):
    imsave('rgb_{}.png'.format(index), cv_image)
    cwd = os.getcwd()
    print("Image saved to {}".format(cwd))
def save_depth_map(depth_img, img_idx):
    depth_map = depth_img * 255
    imsave('depth_{}.png'.format(img_idx), depth_map)

def callback_rgb(ros_img):
    global moving, new_image, img_idx
    if not moving and new_image:    
        cv_image = CvBridge().imgmsg_to_cv2(ros_img, desired_encoding="passthrough")
        print('Saving image {} with size: {}'.format(img_idx, cv_image.shape))

        # np.savetxt("foo{}.csv".format(img_idx), cv_image, delimiter=",")
        save_image(cv_image, img_idx)
        img_idx += 1
        new_image = False
def callback_depth(ros_img):
    global moving, new_image, depth_img_idx
    if not moving and new_image:    
        cv_image = CvBridge().imgmsg_to_cv2(ros_img, desired_encoding="passthrough")
        print('Saving image {} with size: {}'.format(depth_img_idx, cv_image.shape))

        imsave('depth_{}.png'.format(depth_img_idx), cv_image)
        depth_img_idx += 1
        new_image = False

def photo_shooter():
    print("1. Moving to starting position")
    assign_pose_target(0.4, 0.5, 0.6, 0.2, 0.0, 0.0, 0.0)
    print ("Current position 1: {},{},{}".format(group.get_current_pose().pose.position.x,
                                               group.get_current_pose().pose.position.y,
                                               group.get_current_pose().pose.position.z))
    print("2. Moving to position 2")
    assign_pose_target(0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0)
    print ("Current position 2: {},{},{}".format(group.get_current_pose().pose.position.x,
                                               group.get_current_pose().pose.position.y,
                                               group.get_current_pose().pose.position.z))
    global moving, new_image
    rospy.Subscriber('/intel_realsense_camera/rgb/image_raw', Image, callback_rgb)
    rospy.Subscriber('/intel_realsense_camera/depth/image_raw', Image, callback_depth)
    position_y = 0.4
    while not rospy.is_shutdown() and position_y <= 1.0:
        moving = True
        if moving:
            new_image = False
            print("Moving to position: ")
            assign_pose_target(-0.1, position_y, 0.6, 0.0, 0.0, 0.0, 0.0)
            print ("Current position: {},{},{}".format(group.get_current_pose().pose.position.x,
                                            group.get_current_pose().pose.position.y,
                                            group.get_current_pose().pose.position.z))
            position_y += 0.1
            distance, _ = get_distance_gripper_to_object()
            print("Distance: {}".format(distance))
            moving = False
            new_image = True
            time.sleep(1)

    position_x = -1.0
    while not rospy.is_shutdown() and position_x <= 0.4:
        moving = True
        if moving:
            new_image = False
            print("Moving to position: ")
            assign_pose_target(position_x, 0.8, 0.6, 0.0, 0.0, 0.0, 0.0)
            print ("Current position: {},{},{}".format(group.get_current_pose().pose.position.x,
                                            group.get_current_pose().pose.position.y,
                                            group.get_current_pose().pose.position.z))
            position_x += 0.1
            distance, _ = get_distance_gripper_to_object()
            print("Distance: {}".format(distance))
            moving = False
            new_image = True
            time.sleep(1)
'''

###___MAIN___###
if __name__ == '__main__':
    # pub = rospy.Publisher("curr_pose", geometry_msgs.msg.PoseStamped, queue_size=5)
    group.set_end_effector_link('wrist3_Link')

    rospy.Subscriber('/pickbot/target_joint_positions/', JointState, joint_callback)
    rospy.Subscriber('/pickbot/relative_joint_positions', JointState, relative_joint_callback)

    # while not rospy.is_shutdown():
    #   curr_pos = group.get_current_pose()
    #  rate = rospy.Rate(100)
    # pub.publish(curr_pos)
    print("listening to joint states now")
    rospy.spin()