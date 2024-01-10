import math
import os
import numpy as np
import rospy
import time
import sys
import moveit_commander
import moveit_msgs.msg
from tf.transformations import euler_from_quaternion
from openai_ros import robot_gazebo_env
from std_msgs.msg import Empty
from std_msgs.msg import Float64
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import MultiArrayLayout
from std_msgs.msg import MultiArrayDimension
from sensor_msgs.msg import JointState
from sensor_msgs.msg import Image
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import PointCloud2
from gazebo_msgs.msg import LinkStates
from gazebo_msgs.srv import DeleteModel
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Twist
from controller_manager_msgs.srv import SwitchController
from openai_ros.openai_ros_common import ROSLauncher
from pyquaternion import Quaternion
from custom_robot_msgs.msg import Motion
from custom_robot_msgs.msg import Segment
from custom_robot_msgs.msg import Capsule
from openai_ros.robot_envs.modrob_env_moveit import ModRobEnvMoveIt, MoveModrob

class ModRobEnvPathFollowing(robot_gazebo_env.RobotGazeboEnv):
  """Modular robot that uses the failsafe nodelets to perform a path following trajectory planning.
  The main idea is that this class only outputs a goal joint position. 
  The failsafe planning then plans an executable trajectory based on the max vel, acc, and jerk values of the joints.
  The failsafe planning can also be used to ensure safety during the execution of the planed trajectory.
  """

  def __init__(self, ros_ws_abspath):
    """Initialize a new ModRob (modular robot) environment with path following control.
    Puts the robot in an existing simulation world.

    To check any topic we need to have the simulations running, we need to do two things:
    1) Unpause the simulation: without that the stream of data doesn't flow. This is for simulations
    that are pause for whatever the reason
    2) If the simulation was running already for some reason, we need to reset the controllers.
    This has to do with the fact that some plugins with tf, don't understand the reset of the simulation
    and need to be reseted to work properly.

    The Sensors: The sensors accesible are the ones considered usefull for AI learning.

    Sensor Topic List: TODO
    * 

    Actuators Topic List:
    * /' + self.robot_name_ + '/new_goal_motion

    Args:
      * ros_ws_abspath: The absolute path to the catkin_ws
    """
    # The robot name parameter must be in the parameter server
    self.robot_name_ = rospy.get_param('/modrob/robot_name')    
    self.ros_ws_abspath = ros_ws_abspath
    self._init_robot(ros_ws_abspath)

    # Internal Vars
    # Doesnt have any accesibles
    self.controllers_list = []

    # It doesnt use namespace
    self.robot_name_space = ""

    # We launch the init function of the Parent Class robot_gazebo_env.RobotGazeboEnv
    super(ModRobEnvPathFollowing, self).__init__(controllers_list=self.controllers_list,
                                            robot_name_space=self.robot_name_space,
                                            reset_controls=False,
                                            start_init_physics_parameters=False,
                                            reset_world_or_sim="WORLD")
    # Unpause the simulation to initialize all sensors and actors.
    self.gazebo.unpauseSim()
    self.is_collided = False
    self._joint_states_topic = '/' + self.robot_name_ + '/joint_states'
    self._arm_joint_names = rospy.get_param('/' + self.robot_name_ + '/arm_position_controller/joints')

    # We Start all the ROS related Subscribers and publishers
    rospy.Subscriber(self._joint_states_topic, JointState, self._joint_state_callback)
    self._check_all_sensors_ready()
    self._init_joint_publisher()
    self._check_publishers_connection()

    # Load the transformation matrices from the ROS parameter server for forward kinematics.
    self._get_arm_transformation_matrices()

    # Pause the simulation after initialization
    self.gazebo.pauseSim()

    rospy.loginfo("Finished ModRob INIT...")


  def _init_robot(self, ros_ws_abspath):
    """Calls launch file of robot."""
    self._get_base_pose_parameter()
    quat_base = self.convert_pose_to_quaternion(self.base_pose)
    (base_r, base_p, base_y) = euler_from_quaternion([quat_base.w, quat_base.x, quat_base.y, quat_base.z])
    launch_arg_string = "robot_name:={} x:={} y:={} z:={} roll:={} pitch:={} yaw:={}".format(
        self.robot_name_, self.base_pose.position.x, self.base_pose.position.y, self.base_pose.position.z,
        base_r, base_p, base_y
    )
    ROSLauncher(rospackage_name="initialisation",
                launch_file_name="init_modrob.launch",
                ros_ws_abspath=ros_ws_abspath,
                launch_arg_string=launch_arg_string)


  def _get_base_pose_parameter(self):
    """Load base pose from parameter server."""
    # Init robot position
    base_pose_position = rospy.get_param('/modrob/base_pose').get("position")
    base_pose_orientation = rospy.get_param('/modrob/base_pose').get("orientation")
    self.base_pose = self.create_pose([base_pose_position.get("x"), base_pose_position.get("y"), base_pose_position.get("z")], [base_pose_orientation.get("w"), base_pose_orientation.get("x"), base_pose_orientation.get("y"), base_pose_orientation.get("z")])

  def _get_arm_transformation_matrices(self):
    """Read the transformation matrices from the parameter server."""
    tm_vec = rospy.get_param('/' + self.robot_name_ + '/transformation_matrices')
    self.joint_transforamtions = self.vec_to_transformation_matrix(tm_vec)
    tool_tm_vec = rospy.get_param('/' + self.robot_name_ + '/tool_transformation_matrix')
    self.tool_transformation = self.vec_to_transformation_matrix(tool_tm_vec)[0]
    enclosures = rospy.get_param('/' + self.robot_name_ + '/enclosures')
    enclosures = np.reshape(enclosures, [len(self.joint_transforamtions), 7])
    self.capsules = []
    for i in range(len(enclosures)):
      cap = Capsule
      segment = Segment
      segment.p = enclosures[i,0:3]
      segment.q = enclosures[i,3:6]
      cap.segment = segment
      cap.radius = enclosures[i, 6]
      self.capsules.append(cap)

  def _init_joint_publisher(self):
    """Initialize the joint controller publisher with the joint list.
    Relys on joint sensors being published.
    """
    self._arm_joint_publisher = rospy.Publisher('/' + self.robot_name_ + '/new_goal_motion', Motion, queue_size=100)
    self._gripper_joint_publisher = rospy.Publisher('/' + self.robot_name_ + '/gripper_position_controller/command', Float64MultiArray, queue_size=10)

  def _check_all_systems_ready(self):
    """Check that all the sensors, publishers and other simulation systems are operational."""
    self._check_all_sensors_ready()
    return True

  def _check_all_sensors_ready(self):
    """Perform all sensor checks."""
    rospy.logdebug("START ALL SENSORS READY")
    self._check_joint_state_ready()
    rospy.logdebug("ALL SENSORS READY")

  def _check_publishers_connection(self):
    """Check that all the publishers are working"""
    # Check joint position controller publishers
    rate = rospy.Rate(10)  # 10hz
    while self._arm_joint_publisher.get_num_connections() == 0 and not rospy.is_shutdown():
      rospy.logdebug("No susbribers to /" + self.robot_name_ + "/arm_position_controller/command yet so we wait and try again")
      try:
        rate.sleep()
      except rospy.ROSInterruptException:
        # This is to avoid error when world is rested, time when backwards.
        pass
    while self._gripper_joint_publisher.get_num_connections() == 0 and not rospy.is_shutdown():
      rospy.logdebug("No susbribers to /" + self.robot_name_ + "/gripper_position_controller/command yet so we wait and try again")
      try:
        rate.sleep()
      except rospy.ROSInterruptException:
        # This is to avoid error when world is rested, time when backwards.
        pass
    rospy.logdebug("All joint position controller publishers connected!")
    # Check additional publishers
    rospy.logdebug("All publishers READY")

  def _check_joint_state_ready(self):
    self.joint_state = None
    rospy.logdebug("Waiting for {} to be READY...".format(self._joint_states_topic))
    while self.joint_state is None and not rospy.is_shutdown():
      try:
        data = rospy.wait_for_message(self._joint_states_topic, JointState, timeout=5.0)
        if len(data.velocity) > 0:
          self.joint_state = data
        rospy.logdebug("Current {} READY=>".format(self._joint_states_topic))
      except:
        rospy.logerr("Current {} not ready yet, retrying for getting joint states".format(self._joint_states_topic))
    return self.joint_state

  def _joint_state_callback(self, data):
    # Often, there is an empty joint state message.
    if len(data.velocity) > 0:
      self.joint_state = data


  # Methods that the TrainingEnvironment will need.
  # ----------------------------
  def move_arm_joints(self, joint_positions, error=0.05, timeout=3.0):
    """Control the arm joints of the robot.
        The control waits until timeout or desired position reached within a margin of error.
    Args:
        joint_positions (list or np.array): list of desired joint positions
        error (double): absolute error allowed between each desired and reached joint position
        timeout (double): time to wait in s, set this to zero to wait until joint position is reached 
    """
    assert len(self.get_arm_joint_names())==len(joint_positions), "Length of desired arm joint positions does not match"
    # Create motion message
    msg = Motion()
    # Not relevant
    msg.s = 0
    msg.q = joint_positions
    # Goal velocity is zero
    msg.dq = [0 for _ in range(len(joint_positions))]
    # Not relevant
    msg.ddq = [0 for _ in range(len(joint_positions))]
    self._arm_joint_publisher.publish(msg)
    # Wait for a certain amount of time to get to the desired position.
    return self.wait_for_joints_to_get_there(joint_positions, error=error, timeout=timeout)

  def wait_for_joints_to_get_there(self, desired_joint_positions, error=0.05, timeout=3.0):
    """Wait until target joint position is reached within an error or until the timout is reached.
      Set timeout to 0 to wait until joint position is reached.
    Args:
      * desired_joint_positions (list or np.array): list of desired joint positions
      * error (double): absolute error allowed between each desired and reached joint position
      * timeout (double): time to wait in s, set this to zero to wait until joint position is reached 
    """
    assert len(self.get_arm_joint_positions())==len(desired_joint_positions), "Length of desired arm joint positions does not match."
    time_waiting = 0.0
    frequency = 100.0 # Fine tune this parameter.
    are_equal = False
    is_timeout = False
    # ROS will try to keep this frequency of the following loop
    rate = rospy.Rate(frequency)
    rospy.logdebug("Waiting for joint to get to the position")
    success = True
    while not are_equal and not is_timeout and not rospy.is_shutdown():
      are_equal = self.joints_close(self.get_arm_joint_positions(), desired_joint_positions, error)
      rate.sleep()
      time_waiting += 1.0 / frequency
      if timeout > 1e-5:
        is_timeout = time_waiting > timeout
      else:
        if time_waiting > 10:
          # If we waited longer than 10 seconds, reset the robot
          rospy.logerr("ROBOT DID NOT REACHED DESIRED POSITION FOR 10 SECONDS!!! RESETTING ROBOT!")
          self.full_reset_robot()
          success = False
          is_timeout = True
    
    rospy.logdebug("Joints are in the desired position with an error of " + str(error))
    return success


  def full_reset_robot(self):
    """Removes the robot model from simulation and respawns it.
    This is neccesary because the ROS controllers sometimes die.
    """
    # TODO: Test this
    self.gazebo.unpauseSim()
    # rosnode kill /modrob1/joint_state_publisher /modrob1/controller_spawner /modrob1/robot_state_publisher
    os.system("rosnode kill /{}/joint_state_publisher /{}/controller_spawner /{}/robot_state_publisher".format(
        self.robot_name_, self.robot_name_, self.robot_name_))
    # Delte the robot model
    rospy.wait_for_service('/gazebo/delete_model')
    try:
        delete_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
        resp = delete_model(self.robot_name_)
    except rospy.ServiceException as e:
        rospy.logerr("Delete model service call failed: %s"%e)
        resp = False
        return
    # Respawn the robot model
    self._get_base_pose_parameter()
    quat_base = self.convert_pose_to_quaternion(self.base_pose)
    (base_r, base_p, base_y) = euler_from_quaternion([quat_base.w, quat_base.x, quat_base.y, quat_base.z])
    launch_arg_string = "robot_name:={} init_x:={} init_y:={} init_z:={} init_roll:={} init_pitch:={} init_yaw:={}".format(
        self.robot_name_, self.base_pose.position.x, self.base_pose.position.y, self.base_pose.position.z,
        base_r, base_p, base_y
    )
    ROSLauncher(rospackage_name="modrob_simulation",
                launch_file_name="put_robot_in_world_path_following.launch",
                ros_ws_abspath=self.ros_ws_abspath,
                launch_arg_string=launch_arg_string)
    # Wait for all systems (joint state) to be ready
    # TODO: Test this out
    #self._send_initialization()
    self.gazebo.unpauseSim()
    self._check_all_systems_ready()


  def joints_close(self, joint_positions, desired_joint_positions, error):
    """Returns if the joints are in a bound of error to the designated goal."""
    return np.allclose(joint_positions, desired_joint_positions, atol=error)

  def reinit_sensors(self):
      """
      This method is for the tasks so that when reseting the episode
      the sensors values are forced to be updated with the real data and
      <<-- Only needed when reset is set to SIMULATION.
      <<-- TODO: Implement this ?
      """

  def _send_initialization(self):
      """Send init message to ROS topic
      """
      ROSLauncher(rospackage_name="initialisation",
                  launch_file_name="init.launch",
                  ros_ws_abspath=self.ros_ws_abspath)

  # Methods that the TrainingEnvironment will need to define here as virtual
  # because they will be used in RobotGazeboEnv GrandParentClass and defined in the
  # TrainingEnvironment.
  # ----------------------------
  def _set_init_pose(self):
    """Sets the Robot in its init pose"""
    raise NotImplementedError()

  def _init_env_variables(self):
    """Inits variables needed to be initialised each time we reset at the start of an episode."""
    raise NotImplementedError()

  def _compute_reward(self, observations, done):
    """Calculates the reward to give based on the observations given."""
    raise NotImplementedError()

  def _set_action(self, action):
    """Applies the given action to the simulation."""
    raise NotImplementedError()

  def _get_obs(self):
    """Get the observations of this step."""
    raise NotImplementedError()

  def _is_done(self, observations):
    """Checks if episode done based on observations given."""
    raise NotImplementedError()

  def get_arm_joint_positions(self):
    """Return a list of arm joint positions in rad.
    The joint values are in the same oder as get_arm_joint_names()."""
    joint_position_dict = dict(zip(self.joint_state.name, self.joint_state.position))
    return [joint_position_dict.get(joint_name) for joint_name in self.get_arm_joint_names()]

  def get_arm_joint_velocities(self):
    """Return a list of arm joint angular velocities in rad/s.
    The joint values are in the same oder as get_arm_joint_names()."""
    joint_velocity_dict = dict(zip(self.joint_state.name, self.joint_state.velocity))
    return [joint_velocity_dict.get(joint_name) for joint_name in self.get_arm_joint_names()]

  def get_arm_joint_efforts(self):
    """Return a list of arm joint momentum in Nm.
    The joint values are in the same oder as get_arm_joint_names()."""
    joint_effort_dict = dict(zip(self.joint_state.name, self.joint_state.effort))
    return [joint_effort_dict.get(joint_name) for joint_name in self.get_arm_joint_names()]

  def get_arm_joint_names(self):
    """Return list of names in arm joint group."""
    return self._arm_joint_names

  def get_joint_state(self):
    """Return the whole joint state topic dictionary."""
    return self.joint_state

  def get_random_ee_pose(self):
    """Generate a random joint position and return the end effector pose."""
    random_joint_positions = (2 * np.random.rand(len(self._arm_joint_names)) - 1) * math.pi
    ee_pos, ee_quat = self.get_ee_position_and_quaternion(random_joint_positions)
    return self.create_pose(ee_pos, ee_quat), random_joint_positions.tolist()

  def get_random_joint_pose(self, avg_joint_pose, joint_diff):
    """Generate a random joint position and return the end effector pose."""
    random_joint_positions = (np.array(avg_joint_pose) + 
        (2 * np.random.rand(len(self._arm_joint_names)) - 1) * joint_diff)
    ee_pos, ee_quat = self.get_ee_position_and_quaternion(random_joint_positions)
    return self.create_pose(ee_pos, ee_quat), random_joint_positions.tolist()


  def forward_kinematics(self, joint_angles):
    """Calculates the forward kinematics for this robot.
    
    Args: 
      * Joint_angles (list of doubles): Joint angle values in rad

    Returns:
      * Transformation matrix of the end effector
    """
    assert len(self.joint_transforamtions) == len(joint_angles)
    transformation_matrix = np.eye(4)
    for i in range(0, len(self.joint_transforamtions)):
      transformation_matrix = np.matmul(transformation_matrix, self.joint_transforamtions[i])
      transformation_matrix = np.matmul(transformation_matrix, self.get_rot_z_trafo(joint_angles[i]))
    transformation_matrix = np.matmul(transformation_matrix, self.tool_transformation)
    return transformation_matrix

  def get_rot_z_trafo(self, z_rotation):
    """Return a z-rotation matrix with joint angle z_rotation"""
    return np.array([[np.cos(z_rotation), -np.sin(z_rotation), 0, 0], 
                     [np.sin(z_rotation), np.cos(z_rotation), 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])

  def get_ee_pose(self):
    """Return the pose of the end effector."""
    position_vec, orientation_vec = self.get_current_ee_position_and_quaternion()
    return self.create_pose(position_vec, orientation_vec)

  def get_ee_position_and_quaternion(self, joint_values):
    """Return the cartesian position of the end effector and the orientation quaternion"""
    ee_transformation_matrix = self.forward_kinematics(joint_values)
    ee_position = [ee_transformation_matrix[0,3], ee_transformation_matrix[1,3], ee_transformation_matrix[2,3]]
    ee_q = Quaternion(matrix=ee_transformation_matrix[0:3,0:3])
    ee_quaternion = [ee_q.w, ee_q.x, ee_q.y, ee_q.z]
    return ee_position, ee_quaternion

  def check_collision_ground(self, joint_angles, z_ground):
    """Check if any link of the robot collides with the ground.
    
    Args:
      joint_angles (double[n_joints]): angle values of arm joints
      z_ground (double): z-position of the ground
    """
    assert len(self.joint_transforamtions) == len(joint_angles)
    transformation_matrix = np.eye(4)
    for i in range(0, len(self.joint_transforamtions)):
      transformation_matrix = np.matmul(transformation_matrix, self.joint_transforamtions[i])
      transformation_matrix = np.matmul(transformation_matrix, self.get_rot_z_trafo(joint_angles[i]))
      cap_p = np.matmul(transformation_matrix, np.array(np.append(self.capsules[i].segment.p, 1.0)))
      if cap_p[2] < z_ground + self.capsules[i].radius:
        return True
      cap_q = np.matmul(transformation_matrix, np.array(np.append(self.capsules[i].segment.q, 1.0)))
      if cap_q[2] < z_ground + self.capsules[i].radius:
        return True
    return False


  def get_current_ee_position_and_quaternion(self):
    """Return the current cartesian position of the end effector and the orientation quaternion"""
    return self.get_ee_position_and_quaternion(self.get_arm_joint_positions())

  def get_ee_rpy(self):
    """Return the roll, pitch, yaw values of the end effector."""
    raise NotImplementedError()

  def get_ee_quaternion(self):
    """Return the current end effector orientation quaternion (x, y, z, w)."""
    raise NotImplementedError()

  def vec_to_transformation_matrix(self, vec):
    """Convert a vector of numbers to a transformation matrix."""
    assert len(vec)%16 == 0, "Transformation matrix vector has wrong format. len(vec) = {}".format(len(vec))
    n_matrices = int(len(vec)/16)
    list_of_matrices = []
    for i in range(0, n_matrices):
      list_of_matrices.append(np.reshape(vec[i*16:(i+1)*16], [4,4]))
    return list_of_matrices

  def create_pose(self, position_vec, orientation_vec):
    """Create a geometry_msgs.msg.Pose object from position and orientation.
    Args:
        position_vec (list): cartesian position [x, y, z]
        orientation_vec (list): orientation quaternion [w, x, y, z]
    Returns:
        geometry_msgs.msg.Pose
    """
    pose = Pose()
    pose.position.x = position_vec[0]
    pose.position.y = position_vec[1]
    pose.position.z = position_vec[2]
    pose.orientation.w = orientation_vec[0]
    pose.orientation.x = orientation_vec[1]
    pose.orientation.y = orientation_vec[2]
    pose.orientation.z = orientation_vec[3]
    return pose

  def pose_to_vector(self, pose):
    """Convert a pose to a vector containing [pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z]."""
    position = [pose.position.x, pose.position.y, pose.position.z]
    quaternion = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
    return position + quaternion

  def convert_pose_to_quaternion(self, pose):
    """Convert a geometry_msgs.msg.Pose to a pyquaternion.Quaternion.
    TODO: Write utility class and move this to util.
    Args:
        pose (geometry_msgs.msg.Pose)
    Returns:
        pyquaternion.Quaternion
    """
    return Quaternion(w=pose.orientation.w, x=pose.orientation.x, y=pose.orientation.y, z=pose.orientation.z)

  def get_new_x_axis(self, quaternion):
    """Return the new x axis after a quaternion rotation.
    Args:
        quaternion (Quaternion): The quaternion used for rotation
    Returns:
        np.array (shape: [3]): The new x-axis
    """
    return quaternion.rotation_matrix[:,0]
  
  def get_new_y_axis(self, quaternion):
    """Return the new y axis after a quaternion rotation.
    Args:
        quaternion (Quaternion): The quaternion used for rotation
    Returns:
        np.array (shape: [3]): The new y-axis
    """
    return quaternion.rotation_matrix[:,1]

  def get_new_z_axis(self, quaternion):
    """Return the new z axis after a quaternion rotation.
    Args:
        quaternion (Quaternion): The quaternion used for rotation
    Returns:
        np.array (shape: [3]): The new z-axis
    """
    return quaternion.rotation_matrix[:,2]
  
  def great_circle_distance(self, n1, n2):
    """Return the great circle distance between two points on a sphere given by normal vectors.
    See https://en.wikipedia.org/wiki/Great-circle_distance#Vector_version
    Args:
        n1 (np.array, shape: [3]): Normal vector 1
        n2 (np.array, shape: [3]): Normal vector 2
    Returns:
        double: Great circle distance
    """
    return np.arccos(np.dot(n1, n2))
