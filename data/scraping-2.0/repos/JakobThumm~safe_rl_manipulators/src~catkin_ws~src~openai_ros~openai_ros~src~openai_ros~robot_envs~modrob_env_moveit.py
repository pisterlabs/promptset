import numpy as np
import rospy
import time
import sys
import moveit_commander
import moveit_msgs.msg
from tf.transformations import euler_from_quaternion
from openai_ros import robot_gazebo_env
from std_msgs.msg import Float64
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import MultiArrayLayout
from std_msgs.msg import MultiArrayDimension
from sensor_msgs.msg import JointState
from sensor_msgs.msg import Image
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import PointCloud2
from gazebo_msgs.msg import LinkStates
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Twist
from controller_manager_msgs.srv import SwitchController
from openai_ros.openai_ros_common import ROSLauncher
from pyquaternion import Quaternion


class ModRobEnvMoveIt(robot_gazebo_env.RobotGazeboEnv):
    """Superclass for all modular robot environments.
    """

    def __init__(self, ros_ws_abspath):
        """
        Initializes a new ModRob (modular robot) environment with MoveIt! control.
        ModRob doesnt use controller_manager, therefore we wont reset the
        controllers in the standard fashion. For the moment we wont reset them.
        <<--- Can we use the controller manager in the future? --->>

        To check any topic we need to have the simulations running, we need to do two things:
        1) Unpause the simulation: without that the stream of data doesn't flow. This is for simulations
        that are pause for whatever the reason
        2) If the simulation was running already for some reason, we need to reset the controllers.
        This has to do with the fact that some plugins with tf, don't understand the reset of the simulation
        and need to be reseted to work properly.

        The Sensors: The sensors accesible are the ones considered usefull for AI learning.

        Sensor Topic List: TODO
        * 

        Actuators Topic List: TODO
        *

        Args:
        """
        rospy.loginfo("Initialize ModRob environment...")
        # The robot name parameter must be in the parameter server
        self.robot_name_ = rospy.get_param('/modrob/robot_name')  

        # Variables that we give through the constructor.
        # None in this case

        # We launch the ROSlaunch that spawns the robot into the world
        self.ros_ws_abspath = ros_ws_abspath
        self._init_robot(ros_ws_abspath)

        # Internal Vars
        # Doesnt have any accesibles
        self.controllers_list = []

        # It doesnt use namespace
        self.robot_namespace = ""

        # We launch the init function of the Parent Class robot_gazebo_env.RobotGazeboEnv
        super(ModRobEnvMoveIt, self).__init__(controllers_list=self.controllers_list,
                                            robot_namespace=self.robot_namespace,
                                            reset_controls=False,
                                            start_init_physics_parameters=False,
                                            reset_world_or_sim="WORLD")

        self.gazebo.unpauseSim()
        #self.controllers_object.reset_controllers()
        self._joint_states_topic = '/' + self.robot_name_ + '/joint_states'

        # We Start all the ROS related Subscribers and publishers
        rospy.Subscriber(self._joint_states_topic, JointState, self._joint_state_callback)
        self._check_all_sensors_ready()

        ## Controlling the robot
        self.moveit_controllers = ['arm_controller', 'gripper_controller']
        self.ros_controllers = ['arm_position_controller', 'gripper_position_controller']
        # First initialize MoveIt! Service
        self.is_moveit_controller = True
        self.move_object = MoveModrob(joint_states_topic = self._joint_states_topic)
        # Then switch to ROS controller connections
        self._switch_controllers()
        self._init_joint_publisher()
        self._check_publishers_connection()

        self.gazebo.pauseSim()

        rospy.loginfo("Finished ModRob INIT...")

    def _init_robot(self,ros_ws_abspath):
        """Calls launch file of robot."""
        self._get_base_pose_parameter()
        quat_base = self.convert_pose_to_quaternion(self.base_pose)
        (base_r, base_p, base_y) = euler_from_quaternion([quat_base.w, quat_base.x, quat_base.y, quat_base.z])
        launch_arg_string = "robot_name:={} moveit:=true x:={} y:={} z:={} roll:={} pitch:={} yaw:={}".format(
            self.robot_name_, self.base_pose.position.x, self.base_pose.position.y, self.base_pose.position.z,
            base_r, base_p, base_y
        )
        ROSLauncher(rospackage_name="modrob_simulation",
                    launch_file_name="put_robot_in_world.launch",
                    ros_ws_abspath=ros_ws_abspath,
                    launch_arg_string=launch_arg_string)

    def _get_base_pose_parameter(self):
        """Load base pose from parameter server."""
        # Init robot position
        base_pose_position = rospy.get_param('/modrob/base_pose').get("position")
        base_pose_orientation = rospy.get_param('/modrob/base_pose').get("orientation")
        self.base_pose = self.create_pose([base_pose_position.get("x"), base_pose_position.get("y"), base_pose_position.get("z")], 
                                           [base_pose_orientation.get("x"), base_pose_orientation.get("y"), base_pose_orientation.get("z"), base_pose_orientation.get("w")])
        

    # Methods needed by the RobotGazeboEnv
    # ----------------------------
    def _switch_controllers(self):
        """Switch between moveit and ROS controllers."""
        if self.is_moveit_controller:
            # Switch to ROS controllers
            start_controllers = self.ros_controllers
            stop_controllers = self.moveit_controllers
        else:
            # Switch to moveit control
            start_controllers = self.moveit_controllers
            stop_controllers = self.ros_controllers
            
        rospy.wait_for_service('/' + self.robot_name_ + '/controller_manager/switch_controller')
        try:
            switch_controller_call = rospy.ServiceProxy('/' + self.robot_name_ + '/controller_manager/switch_controller', SwitchController)
            switch_controller_call(start_controllers = start_controllers,
                                    stop_controllers = stop_controllers,
                                    strictness = 2,
                                    start_asap = True)
            # Switch boolean
            self.is_moveit_controller = not self.is_moveit_controller
            rospy.loginfo("Switched controllers from {} to {}".format(stop_controllers, start_controllers))
            """
            switch_msg = SwitchController()
            switch_msg.start_controllers = ['arm_position_controller', 'gripper_position_controller']
            switch_msg.stop_controllers = ['arm_controller', 'gripper_controller']
            switch_msg.strictness = 2
            switch_msg.start_asap = True
            """
        except rospy.ServiceException as e:
            rospy.logerr("Switch controllers service call failed: %s"%e)

    def _init_joint_publisher(self):
        """Initialize the joint controller publisher with the joint list.
        Relys on joint sensors being published.
        """
        self._arm_joint_publisher = rospy.Publisher('/' + self.robot_name_ + '/arm_position_controller/command', Float64MultiArray, queue_size=10)
        self._gripper_joint_publisher = rospy.Publisher('/' + self.robot_name_ + '/gripper_position_controller/command', Float64MultiArray, queue_size=10)

    def _check_all_systems_ready(self):
        """
        Checks that all the sensors, publishers and other simulation systems are
        operational.
        """
        self._check_all_sensors_ready()
        return True

    def _check_all_sensors_ready(self):
        rospy.logdebug("START ALL SENSORS READY")
        self._check_joint_state_ready()
        rospy.logdebug("ALL SENSORS READY")

    def _check_publishers_connection(self):
        """Checks that all the publishers are working.
        """
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
                self.joint_state = rospy.wait_for_message(self._joint_states_topic, JointState, timeout=5.0)
                rospy.logdebug("Current {} READY=>".format(self._joint_states_topic))

            except:
                rospy.logerr("Current {} not ready yet, retrying for getting joint states".format(self._joint_states_topic))

        return self.joint_state

    def _joint_state_callback(self, data):
        # Often, there is an empty joint state message.
        if len(data.velocity) > 0:
            self.joint_state = data

    # Methods that the TrainingEnvironment will need to define here as virtual
    # because they will be used in RobotGazeboEnv GrandParentClass and defined in the
    # TrainingEnvironment.
    # ----------------------------
    def _set_init_pose(self):
        """Sets the Robot in its init pose
        """
        raise NotImplementedError()

    def _init_env_variables(self):
        """Inits variables needed to be initialised each time we reset at the start
        of an episode.
        """
        raise NotImplementedError()

    def _compute_reward(self, observations, done):
        """Calculates the reward to give based on the observations given.
        """
        raise NotImplementedError()

    def _set_action(self, action):
        """Applies the given action to the simulation.
        """
        raise NotImplementedError()

    def _get_obs(self):
        raise NotImplementedError()

    def _is_done(self, observations):
        """Checks if episode done based on observations given.
        """
        raise NotImplementedError()

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
        assert len(self.get_arm_joint_positions())==len(joint_positions), "Length of desired arm joint positions does not match."
        if self.is_moveit_controller:
            result = self.move_object.move_joints(joint_positions)
            if not result:
                rospy.logwarn("Cannot execute path to joint positions: {}!".format(joint_positions))
        else:
            msg_dim = [MultiArrayDimension(label="joint_positions", size = len(joint_positions), stride = len(joint_positions))]
            msg_layout = MultiArrayLayout(dim = msg_dim, data_offset = 0)
            msg = Float64MultiArray(layout = msg_layout, data=joint_positions)
            self._arm_joint_publisher.publish(msg)
        # Wait for a certain amount of time to get to the desired position.
        self.wait_for_joints_to_get_there(joint_positions, error=error, timeout=timeout)

    def wait_for_joints_to_get_there(self, desired_joint_positions, error=0.05, timeout=3.0):
        """Wait until target joint position is reached within an error or
            until the timout is reached.
            Set timeout to 0 to wait until joint position is reached.
        Args:
            desired_joint_positions (list or np.array): list of desired joint positions
            error (double): absolute error allowed between each desired and reached joint position
            timeout (double): time to wait in s, set this to zero to wait until joint position is reached 
        """
        assert len(self.get_arm_joint_positions())==len(desired_joint_positions), "Length of desired arm joint positions does not match."
        time_waiting = 0.0
        frequency = 100.0 # Fine tune this parameter.
        are_equal = False
        is_timeout = False
        # ROS will try to keep this frequency of the following loop
        rate = rospy.Rate(frequency)
        rospy.logdebug("Waiting for joint to get to the position")
        while not are_equal and not is_timeout and not rospy.is_shutdown():
            are_equal = np.allclose(self.get_arm_joint_positions(), desired_joint_positions, atol=error)
            rate.sleep()
            if timeout > 1e-5:
                time_waiting += 1.0 / frequency
                is_timeout = time_waiting > timeout

        rospy.logdebug("Joints are in the desired position with an error of " + str(error))

    def move_arm_ee(self, ee_pose, error=0.05, timeout=3.0):
        """Control the arm joints of the robot.
            The control waits until timeout or desired position reached within a margin of error.
            The control automatically switches to moveit control if it is in ROS control mode before and switches back afterwards.
        Args:
            ee_pose (geometry_msgs.msg.Pose): desired end effector pose
            error (double): absolute error allowed between each desired and reached end effector pose
            timeout (double): time to wait in s, set this to zero to wait until joint position is reached 
        """
        assert isinstance(ee_pose, Pose), "ee_pose is not of type geometry_msgs.msg.Pose!"
        # We need moveit control to move to an ee pose.
        controllers_switched = False
        if not self.is_moveit_controller:
            self._switch_controllers()
            controllers_switched = True
        result = self.move_object.move_ee(ee_pose)
        if not result:
            rospy.logwarn("Cannot execute path to ee pose: {}!".format(ee_pose))
        # Wait for a certain amount of time to get to the desired position.
        self.wait_for_ee_to_get_there(ee_pose, error=error, timeout=timeout)
        # Switch back to ROS control.
        if controllers_switched:
            self._switch_controllers()

    def wait_for_ee_to_get_there(self, desired_ee_pose, error=0.05, timeout=3.0):
        """Wait until target end effector pose is reached within an error or
            until the timout is reached.
            Set timeout to 0 to wait until joint position is reached.
        Args:
            desired_ee_pose (geometry_msgs.msg.Pose): desired end effector pose
            error (double): absolute error allowed between each desired and reached end effector pose.
                            The error is both for cartesian 3D distance and orientation distance. 
                            (Maybe separate into two if necessary)
            timeout (double): time to wait in s, set this to zero to wait until joint position is reached 
        """
        assert isinstance(desired_ee_pose, Pose), "desired_ee_pose is not of type geometry_msgs.msg.Pose!"
        time_waiting = 0.0
        frequency = 100.0 # Fine tune this parameter.
        are_equal = False
        is_timeout = False
        # ROS will try to keep this frequency of the following loop
        rate = rospy.Rate(frequency)
        rospy.logdebug("Waiting for joint to get to the position")
        while not are_equal and not is_timeout and not rospy.is_shutdown():
            cartesian_distance = self.move_object.calculate_ee_cartesian_distance(desired_ee_pose)
            orientation_distance = self.move_object.calculate_ee_orientation_distance(desired_ee_pose)
            are_equal = (cartesian_distance <= error) and (orientation_distance <= error)
            rate.sleep()
            if timeout == 0.0:
                # Dismiss time constraint and wait until target reached
                time_waiting += 0.0
            else:
                time_waiting += 1.0 / frequency
            is_timeout = time_waiting > timeout
            
        rospy.logdebug("Joints are in the desired position with an erro of " + str(error))

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
        return self.move_object.get_arm_joints()

    def get_joint_state(self):
        """Return the whole joint state topic dictionary (not recommended for moveit usage)."""
        return self.joint_state

    def get_ee_pose(self):
        """Return the pose of the end effector."""
        return self.move_object.get_ee_pose()

    def get_ee_position(self):
        """Return the cartesian position of the end effector."""
        return self.move_object.get_ee_position()

    def get_ee_rpy(self):
        """Return the roll, pitch, yaw values of the end effector."""
        return self.move_object.get_ee_rpy()

    def get_ee_quaternion(self):
        """Return the current end effector orientation quaternion (x, y, z, w)."""
        return self.move_object.get_ee_quaternion()

    def reinit_sensors(self):
        """
        This method is for the tasks so that when reseting the episode
        the sensors values are forced to be updated with the real data and
        <<-- Only needed when reset is set to SIMULATION.
        <<-- TODO: Implement this ?
        """

    def create_pose(self, position_vec, orientation_vec):
        """Create a geometry_msgs.msg.Pose object from position and orientation.
        Args:
            position_vec (list): cartesian position [x, y, z]
            orientation_vec (list): orientation quaternion [x, y, z, w]
        Returns:
            geometry_msgs.msg.Pose
        """
        pose = Pose()
        pose.position.x = position_vec[0]
        pose.position.y = position_vec[1]
        pose.position.z = position_vec[2]
        pose.orientation.x = orientation_vec[0]
        pose.orientation.y = orientation_vec[1]
        pose.orientation.z = orientation_vec[2]
        pose.orientation.w = orientation_vec[3]
        return pose

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

class MoveModrob(object):
    """Class for communicating with MoveIt!
    
    There are 2 types of goal targets:

    a JointValueTarget (aka JointStateTarget) specifies an absolute value for each joint (angle for rotational joints or position for prismatic joints).
    a PoseTarget (Position, Orientation, or Pose) specifies the pose of one or more end effectors (and the planner can use any joint values that reaches the pose(s)).
    See here (http://docs.ros.org/en/diamondback/api/geometry_msgs/html/msg/Pose.html) for ee pose definition.
    """

    def __init__(self, joint_states_topic="/joint_states"):
        rospy.logdebug("===== In MoveModrob")
        init_joint_state_topic = ['joint_states:={}'.format(joint_states_topic)]
        moveit_commander.roscpp_initialize(init_joint_state_topic)
        moveit_commander.roscpp_initialize(sys.argv)
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.arm_group = moveit_commander.MoveGroupCommander("arm")
        self.gripper_group = moveit_commander.MoveGroupCommander("gripper")
        rospy.logdebug("===== Out MoveModrob")

    def move_ee(self, pose_target):
        """Set a new end effector target pose and move to target.
        Args:
            pose_target (geometry_msgs.msg.Pose): a PoseTarget (Position, Orientation, or Pose) for the end effector
                example: pose_target = geometry_msgs.msg.Pose()
                         pose_target.orientation.w = 1.0
                         pose_target.position.x = 0.4
                         pose_target.position.y = 0.1
                         pose_target.position.z = 0.4
        Returns:
            result (Boolean): Whether trajectory is executable
        """
        assert isinstance(pose_target, Pose), "pose_target is not of type geometry_msgs.msg.Pose!"
        self.arm_group.set_pose_target(pose_target)
        result = self.execute_trajectory()
        return result

    def move_joints(self, joint_value_target):
        """Set a new joint value target and move to target.
        Args:
            joint_value_target (list): a JointValueTarget (aka JointStateTarget) for the joints of the arm.
        Returns:
            result (Boolean): Whether trajectory is executable
        """
        self.arm_group.set_joint_value_target(joint_value_target)
        result = self.execute_trajectory()
        return result

    def execute_trajectory(self):
        """Plan a path to the previously set goal position and execute it.
        Note: the robot’s current joint state must be within some tolerance of the first waypoint in the RobotTrajectory or execute() will fail.
        Returns:
            result (Boolean): Whether trajectory is executable
        """
        self.plan = self.arm_group.plan()
        result = self.arm_group.go(wait=False)
        return result

    def stop_execution(self):
        """Stop the arm movement."""
        self.arm_group.stop()

    def clear_all_targets(self):
        """Clear all targets for the arm movement."""
        self.arm_group.clear_pose_targets()

    def get_ee_pose(self):
        """Return the current end effector pose."""
        return self.arm_group.get_current_pose().pose

    def get_ee_position(self):
        """Return the cartesian position of the end effector."""
        ee_position = self.get_ee_pose().position
        return [ee_position.x, ee_position.y, ee_position.z]

    def get_ee_quaternion(self):
        """Return the current end effector orientation quaternion (x, y, z, w)."""
        ee_orientation = self.get_ee_pose().orientation
        return [ee_orientation.x, ee_orientation.y, ee_orientation.z, ee_orientation.w]

    def get_ee_rpy(self):
        """Return the current end effector roll pitch yaw values."""
        return self.arm_group.get_current_rpy()

    def get_current_arm_joint_values(self):
        """Return the current arm joint positions."""
        return self.arm_group.get_current_joint_values()

    def get_current_gripper_joint_values(self):
        """Return the current arm joint positions."""
        return self.gripper_group.get_current_joint_values()

    def get_arm_joints(self):
        """Return list of names of joints in robot arm group."""
        return self.arm_group.get_active_joints()

    def get_gripper_joints(self):
        """Return list of names of joints in gripper group."""
        return self.gripper_group.get_active_joints()

    def calculate_ee_cartesian_distance(self, pose_target):
        """Calculate cartesian position distance between current end effector and given goal pose.
        Args:
            pose_target (geometry_msgs.msg.Pose)
        Returns:
            Cartesian distance in meter.
        """
        assert isinstance(pose_target, Pose), "pose_target is not of type geometry_msgs.msg.Pose!"
        ee_position = self.get_ee_pose().position
        return np.sqrt((ee_position.x - pose_target.x)**2 + 
                       (ee_position.y - pose_target.y)**2 + 
                       (ee_position.z - pose_target.z)**2)
    
    def calculate_ee_orientation_distance(self, pose_target):
        """Calculate distance between the current end effector pose quaternion and the quaternion given in pose_target.
        We use http://kieranwynn.github.io/pyquaternion/#distance-computation for the quaternion distance calculation.
        Note: This function does not measure the distance on the hypersphere, but it takes into account the fact that q and -q encode the same rotation. 
        It is thus a good indicator for rotation similarities.
        Args:
            pose_target (geometry_msgs.msg.Pose)
        Returns:
            Distance between the two quaternions.
        """
        assert isinstance(pose_target, Pose), "pose_target is not of type geometry_msgs.msg.Pose!"
        ee_orientation = self.get_ee_pose().orientation
        q0 = self.convert_pose_to_quaternion(ee_orientation)
        q1 = self.convert_pose_to_quaternion(pose_target)
        return Quaternion.absolute_distance(q0, q1)
    
    def convert_pose_to_quaternion(self, pose):
        """Convert a geometry_msgs.msg.Pose to a pyquaternion.Quaternion.
        TODO: Write utility class and move this to util.
        """
        return Quaternion(w=pose.w, x=pose.x, y=pose.y, z=pose.z)
