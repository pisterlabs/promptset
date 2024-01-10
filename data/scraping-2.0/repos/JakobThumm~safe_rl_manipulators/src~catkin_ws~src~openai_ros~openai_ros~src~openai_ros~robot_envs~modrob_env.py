import numpy
import rospy
import time
from openai_ros import robot_gazebo_env
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState
from sensor_msgs.msg import Image
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import PointCloud2
from gazebo_msgs.msg import LinkStates
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from openai_ros.openai_ros_common import ROSLauncher


class ModRobEnv(robot_gazebo_env.RobotGazeboEnv):
    """Superclass for all modular robot environments.
    """

    def __init__(self, ros_ws_abspath):
        """
        Initializes a new ModRob (modular robot) environment.
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
        # The robot name parameter must be in the parameter server
        self.robot_name_ = rospy.get_param('/robot_name')  

        rospy.loginfo("Initialize ModRob environment...")
        # Variables that we give through the constructor.
        # None in this case

        # We launch the ROSlaunch that spawns the robot into the world
        ROSLauncher(rospackage_name="modrob_simulation",
                    launch_file_name="put_robot_in_world.launch",
                    ros_ws_abspath=ros_ws_abspath)

        # Internal Vars
        # Doesnt have any accesibles
        self.controllers_list = []

        # It doesnt use namespace
        self.robot_name_space = ""

        # We launch the init function of the Parent Class robot_gazebo_env.RobotGazeboEnv
        super(ModRobEnv, self).__init__(controllers_list=self.controllers_list,
                                            robot_name_space=self.robot_name_space,
                                            reset_controls=False,
                                            start_init_physics_parameters=False,
                                            reset_world_or_sim="WORLD")

        self.gazebo.unpauseSim()
        #self.controllers_object.reset_controllers()
        

        # We Start all the ROS related Subscribers and publishers
        rospy.Subscriber("/" + self.robot_name_ + "/joint_states", JointState, self._joint_state_callback)
        self._check_all_sensors_ready()
        #self._cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self._init_joint_publisher()
        self._check_publishers_connection()

        self.gazebo.pauseSim()

        rospy.loginfo("Finished ModRob INIT...")

    # Methods needed by the RobotGazeboEnv
    # ----------------------------

    def _init_joint_publisher(self):
        """Initialize the joint controller publisher with the joint list.
        Relys on joint sensors being published.
        """
        self._check_joint_state_ready()
        assert len(self.joint_state.name) > 0, "No joint names found in joint_state."
        self._joint_publishers = dict()
        for joint_name in self.joint_state.name:
            self._joint_publishers[joint_name] = rospy.Publisher('/' + self.robot_name_ + '/{}_position_controller/command'.format(joint_name), Float64, queue_size=10)

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

    def _check_joint_state_ready(self):
        self.joint_state = None
        rospy.logdebug("Waiting for /" + self.robot_name_ + "/joint_states to be READY...")
        while self.joint_state is None and not rospy.is_shutdown():
            try:
                self.joint_state = rospy.wait_for_message("/" + self.robot_name_ + "/joint_states", JointState, timeout=5.0)
                rospy.logdebug("Current /" + self.robot_name_ + "/joint_states READY=>")

            except:
                rospy.logerr("Current /" + self.robot_name_ + "/joint_states not ready yet, retrying for getting joint states")

        return self.joint_state

    def _joint_state_callback(self, data):
        self.joint_state = data

    def _check_publishers_connection(self):
        """Checks that all the publishers are working.
        """
        # Check joint position controller publishers
        rate = rospy.Rate(10)  # 10hz
        for joint_name in self._joint_publishers:
            publisher = self._joint_publishers[joint_name]
            while publisher.get_num_connections() == 0 and not rospy.is_shutdown():
                rospy.logdebug("No susbribers to /" + self.robot_name_ + "/{}_position_controller/command yet so we wait and try again".format(joint_name))
                try:
                    rate.sleep()
                except rospy.ROSInterruptException:
                    # This is to avoid error when world is rested, time when backwards.
                    pass
        rospy.logdebug("All joint position controller publishers connected!")
        # Check additional publishers
        rospy.logdebug("All publishers READY")

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
    def move_all_joints(self, joint_positions, error=0.2, timeout=3.0):
        """Control the joints and gripper of modrob0.
            The control waits until timeout or desired position reached within a margin of error.
        Args:
            joint_positions (Dict): key: joint_name, value: desired joint position
            error (double): combined absolute error allowed between desired and reached joint position
            timeout (double): time to wait in s, set this to zero to wait until joint position is reached 
        """
        # Check if publishers are active
        self._check_publishers_connection()
        # Send control command to all joints in joint_position dict
        for joint_name in joint_positions:
            if joint_name in self._joint_publishers:
                # Publish val = joint_positions[joint_name] to joint publisher self._joint_publishers[joint_name]
                self._joint_publishers[joint_name].publish(joint_positions[joint_name])
            else:
                rospy.logwarn("Joint /" + self.robot_name_ + "/{}_position_controller/command not found! Not publishing this joint position.".format(joint_name))
        # Wait for a certain amount of time to get to the desired position.
        self.wait_for_joints_to_get_there(joint_positions, error=error, timeout=timeout)

    def wait_for_joints_to_get_there(self, desired_joint_positions, error=0.2, timeout=3.0):
        """Wait until target joint position is reached within an error or
            until the timout is reached.
            Set timeout to 0 to wait until joint position is reached.
        Args:
            desired_joint_positions (Dict): key: joint_name, value: desired joint position
            error (double): combined absolute error allowed between desired and reached joint position
            timeout (double): time to wait in s, set this to zero to wait until joint position is reached 
        """
        time_waiting = 0.0
        frequency = 100.0 # Fine tune this parameter.
        are_equal = False
        is_timeout = False
        # ROS will try to keep this frequency of the following loop
        rate = rospy.Rate(frequency)
        rospy.logdebug("Waiting for joint to get to the position")
        while not are_equal and not is_timeout and not rospy.is_shutdown():

            current_joint_positions = self.get_joint_positions()
            sum_distance = 0
            for joint_name in desired_joint_positions:
                if joint_name in current_joint_positions:
                    # TODO: Handle gripper position different. 
                    # We are currently adding angle differences and cartesian differences
                    sum_distance += abs(desired_joint_positions[joint_name] - current_joint_positions[joint_name])
                else:
                    rospy.logwarn("Joint /" + self.robot_name_ + "/{}_position_controller/command not found! Not checking this joint position.".format(joint_name))
            
            are_equal = sum_distance <= error
            rate.sleep()
            if timeout == 0.0:
                # Dismiss time constraint and wait until target reached
                time_waiting += 0.0
            else:
                time_waiting += 1.0 / frequency
            is_timeout = time_waiting > timeout

        rospy.logdebug(
            "Joints are in the desired position with an erro of "+str(error))

    def get_joint_positions(self):
        return dict(zip(self.joint_state.name, self.joint_state.position))

    def get_joint_velocities(self):
        return dict(zip(self.joint_state.name, self.joint_state.velocity))

    def get_joint_efforts(self):
        return dict(zip(self.joint_state.name, self.joint_state.effort))

    def get_joint_names(self):
        return self.joint_state.name

    def get_joint_state(self):
        return self.joint_state

    def reinit_sensors(self):
        """
        This method is for the tasks so that when reseting the episode
        the sensors values are forced to be updated with the real data and
        <<-- Only needed when reset is set to SIMULATION.
        <<-- TODO: Implement this ?
        """

