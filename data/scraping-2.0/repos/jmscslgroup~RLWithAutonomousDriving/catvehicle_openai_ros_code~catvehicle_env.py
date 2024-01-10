import rospy
import time
import numpy as np
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64, Float32
from openai_ros import robot_gazebo_env
from nav_msgs.msg import Odometry


class CATVehicleEnv(robot_gazebo_env.RobotGazeboEnv):

    def __init__(self):
        
        """
        Initializes a new CATVehicle environment.
        CATVehicle doesnt use controller_manager, therefore we wont reset the 
        controllers in the standard fashion. For the moment we wont reset them.
        
        To check any topic we need to have the simulations running, we need to do two things:
        1) Unpause the simulation: without that the stream of data doesnt flow. This is for simulations
        that are pause for whatever the reason
        2) If the simulation was running already for some reason, we need to reset the controlers.
        This has to do with the fact that some plugins with tf, dont understand the reset of the simulation
        and need to be reseted to work properly.
        """
        
        rospy.logdebug("Start CATVehicle_ENV INIT...")
        
        self.controllers_list = []
        self.publishers_array = []
        self.robot_name_space = ""
        self.reset_controls = False

        
        
        # We launch the init function of the Parent Class robot_gazebo_env.RobotGazeboEnv
        super(CATVehicleEnv, self).__init__(controllers_list=self.controllers_list,
                                            robot_name_space=self.robot_name_space,
                                            reset_controls=False,
                                            start_init_physics_parameters=False,
                                            reset_world_or_sim="WORLD")
        
        self.gazebo.unpauseSim()
        self._check_all_sensors_ready()
        
        self._cmd_vel_pub = rospy.Publisher('/catvehicle/cmd_vel', Twist, queue_size=10)
        
        rospy.Subscriber("/catvehicle/distanceEstimatorSteeringBased/dist", Float64, self._distsb_callback)
        rospy.Subscriber("/catvehicle/distanceEstimatorSteeringBased/angle", Float64, self._anglesb_callback)
        rospy.Subscriber("/catvehicle/distanceEstimator/dist", Float32, self._dist_callback)
        rospy.Subscriber("/catvehicle/distanceEstimator/angle", Float32, self._angle_callback)
        rospy.Subscriber("/catvehicle/odom", Odometry, self._odom_callback)
                
        self._check_publishers_connection()
        self.gazebo.pauseSim()
        
        rospy.logdebug("Finished TurtleBot2Env INIT...")
        
        
    # Methods needed by the RobotGazeboEnv
    # ----------------------------
    
    def _check_all_systems_ready(self):
        """
        Checks that all the sensors, publishers and other simulation systems are
        operational.
        """
        
        self._check_all_sensors_ready()
        #self._check_joint_states_ready()
        self._check_cmd_vel_pub()
        
        return True

    def _check_all_sensors_ready(self):
        """
        Checks that all the sensors, publishers and other simulation systems are
        operational.
        """
        
        self._check_dist_ready()
        self._check_angle_ready()
        self._check_odom_ready()
        self._check_distsb_ready()
        self._check_anglesb_ready()
        
        return True
        
    # Check our distance sensor working
    def _check_dist_ready(self):
        self.dist = None
        rospy.logdebug("Waiting for /catvehicle/distanceEstimator/dist to be READY...")
        while self.dist is None and not rospy.is_shutdown():
            try:
                self.dist = rospy.wait_for_message("/catvehicle/distanceEstimator/dist", Float32, timeout=5.0)
                rospy.logdebug("Current /catvehicle/distanceEstimator/dist READY=>")

            except:
                rospy.logerr("Current /catvehicle/distanceEstimator/dist not ready yet, retrying for getting dist")
        return self.dist
        
    # Checks our angle sensor is working
    def _check_angle_ready(self):
        self.angle = None
        rospy.logdebug("Waiting for /catvehicle/distanceEstimator/angle to be READY...")
        while self.angle is None and not rospy.is_shutdown():
            try:
                self.angle = rospy.wait_for_message("/catvehicle/distanceEstimator/angle", Float32, timeout=5.0)
                rospy.logdebug("Current /catvehicle/distanceEstimator/angle READY=>")

            except:
                rospy.logerr("Current /catvehicle/distanceEstimator/angle not ready yet, retrying for getting angle")
        return self.angle
                
    def _check_cmd_vel_pub(self):
        rate = rospy.Rate(10)  # 10hz
        while self._cmd_vel_pub.get_num_connections() == 0 and not rospy.is_shutdown():
            rospy.logdebug("No susbribers to _cmd_vel_pub yet so we wait and try again")
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                # This is to avoid error when world is rested, time when backwards.
                pass
        rospy.logdebug("_cmd_vel_pub Publisher Connected")
                
     # Check that all publishers are working
    def _check_publishers_connection(self):
        """
        Checks that all the publishers are working
        :return:
        """
        
        rate = rospy.Rate(10)  # 10hz; HOW DOES THIS WORK FOR CATVEHICLE/
        
        self._check_cmd_vel_pub()

        rospy.logdebug("All Publishers READY")
        
    def _check_odom_ready(self):
        self.odom = None
        rospy.logdebug("Waiting for /catvehicle/odom to be READY...")
        while self.dist is None and not rospy.is_shutdown():
            try:
                self.dist = rospy.wait_for_message("/catvehicle/odom", Odometry, timeout=5.0)
                rospy.logdebug("Current /catvehicle/odom READY=>")

            except:
                rospy.logerr("Current /catvehicle/odom not ready yet, retrying for getting odom")

        return self.dist
    
    def _check_distsb_ready(self):
        self.distsb = None
        rospy.logdebug("Waiting for /catvehicle/distanceEstimatorSteeringBased/dist to be READY...")
        while self.distsb is None and not rospy.is_shutdown():
            try:
                self.distsb = rospy.wait_for_message("/catvehicle/distanceEstimatorSteeringBased/dist", Float64, timeout=5.0)
                rospy.logdebug("Current /catvehicle/distanceEstimatorSteeringBased/dist READY=>")

            except:
                rospy.logerr("Current /catvehicle/distanceEstimatorSteeringBased/dist not ready yet, retrying for getting dist")

        return self.distsb
        
    def _check_anglesb_ready(self):
        self.anglesb = None
        rospy.logdebug("Waiting for /catvehicle/distanceEstimatorSteeringBased/angle to be READY...")
        while self.anglesb is None and not rospy.is_shutdown():
            try:
                self.anglesb = rospy.wait_for_message("/catvehicle/distanceEstimatorSteeringBased/angle", Float64, timeout=5.0)
                rospy.logdebug("Current /catvehicle/distanceEstimatorSteeringBased/angle READY=>")

            except:
                rospy.logerr("Current /catvehicle/distanceEstimatorSteeringBased/angle not ready yet, retrying for getting angle")

        return self.anglesb
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
    
    def move_car(self, linear_speed, angular_speed, epsilon=0.05, update_rate=10, min_laser_distance=-1):
        """
        It will move the car based on the linear and angular speeds given.
        (no) It will wait untill those twists are achived reading from the odometry topic.
        :param linear_speed: Speed in the X axis of the robot base frame
        :param angular_speed: Speed of the angular turning of the robot base frame
        :param epsilon: Acceptable difference between the speed asked and the odometry readings
        :param update_rate: Rate at which we check the odometry.
        :return: 
        """
        cmd_vel_value = Twist() # Describes linear motion and angular motion of robot
        cmd_vel_value.linear.x = linear_speed
        cmd_vel_value.angular.z = angular_speed
        rospy.logwarn("CATVehicle Base Twist Cmd>>" + str(cmd_vel_value))
        self._check_publishers_connection()
        self._cmd_vel_pub.publish(cmd_vel_value)
        time.sleep(0.01) # This is the timespan per timestep?
        #time.sleep(0.02)
        
                """ # Implement this later?
        self.wait_until_twist_achieved(cmd_vel_value,
                                        epsilon,
                                        update_rate,
                                        min_laser_distance)
        """
        
    def has_crashed(self, min_distance):
        """
        It states based on the laser scan if the robot has crashed or not.
        Crashed means that the minimum laser reading is lower than the
        min_laser_distance value given.
        If min_laser_distance == -1, it returns always false, because its the way
        to deactivate this check.
        """
        robot_has_crashed = False
        dist = self.distsb.data

        if (dist <= min_distance):
            rospy.logwarn("CATVehicle HAS CRASHED >>> item = " + str(dist)+" < "+str(min_distance))
            robot_has_crashed = True
        return robot_has_crashed
        
    def get_dist(self):
        return self.dist
        
    def get_angle(self):
        return self.angle
        
    def _dist_callback(self, data):
        self.dist = data
    
    def _angle_callback(self, data):
        self.angle = data
        
    def _distsb_callback(self, data):
        self.distsb = data
        
    def _anglesb_callback(self, data):
        self.anglesb = data
        
    def _odom_callback(self, data):
        self.odom = data
    
    
