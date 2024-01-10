#!/usr/bin/env python
import os
import subprocess32 as subprocess
import rospy
import time
import tf
import mavros
import rospkg
import openai_ros.robot_gazebo_env
from mavros_msgs.msg import State, ParamValue
from sensor_msgs.msg import NavSatFix
from mavros_msgs.srv import ParamSet, ParamGet, SetMode, CommandBool, CommandBoolRequest, CommandTOL
from geometry_msgs.msg import PoseStamped, TwistStamped, Quaternion
from openai_ros.roslauncher import ROSLauncher
from openai_ros import robot_gazebo_env



class MavDroneEnv(robot_gazebo_env.RobotGazeboEnv):
    """Superclass for all PX4 MavDrone environments.
    """
    def __init__(self):
        """Initializes a new MavROS environment. \\
        To check the ROS topics, unpause the paused simulation \\
        or reset the controllers if simulation is running.

        Sensors for RL observation space (by topic list): \\
        
        * /mavros/local_position/pose                   
        * /mavros/local_position/velocity_body          
        }


        Actuations for RL action space (by topic list):
        * /cmd_vel: Move the Drone Around when you have taken off.

        Args:
        """


        rospy.logdebug("Start MavDroneEnv INIT...")
        # Variables that we give through the constructor.

        # Internal Vars
        self.controllers_list = ['my_robot_controller1, my_robot_controller2 , ..., my_robot_controllerX']

        self.robot_name_space = ""
        self.px4_ekf2_path = os.path.join(rospkg.RosPack().get_path("px4"),"build/px4_sitl_default/bin/px4-ekf2")


        #reset_controls_bool = True or False
        
        # We launch the init function of the Parent Class robot_gazebo_env.RobotGazeboEnv
        
        super(MavDroneEnv, self).__init__(controllers_list=self.controllers_list,
                                             robot_name_space=self.robot_name_space,
                                             reset_controls=False,
                                             start_init_physics_parameters=True,
                                             reset_world_or_sim="WORLD")
        self.gazebo.unpauseSim()

        # self.ros_launcher = ROSLauncher(rospackage_name="mavros_moveit", launch_file_name="posix_sitl_2.launch")
                    
        self._current_pose = PoseStamped()
        self._current_state= State()
        self._current_gps  = NavSatFix()

        self._check_all_sensors_ready()

        # We Start all the ROS related Subscribers and publishers
        rospy.Subscriber('mavros/state', State, callback=self._stateCb, queue_size=10)
        rospy.Subscriber('mavros/local_position/pose', PoseStamped , callback=self._poseCb, queue_size=10)
        rospy.Subscriber('/mavros/global_position/raw/fix', NavSatFix, callback=self._gpsCb, queue_size=10)
        

        # setpoint publishing rate must be faster than 2Hz. From MavROS documentation
        self._rate = rospy.Rate(20.0)

        self._local_vel_pub = rospy.Publisher('mavros/setpoint_velocity/cmd_vel',TwistStamped,  queue_size=10)

        
        self._arming_client = rospy.ServiceProxy('mavros/cmd/arming',CommandBool) #mavros service for arming/disarming the robot
        self._set_mode_client = rospy.ServiceProxy('mavros/set_mode', SetMode) #mavros service for setting mode. Position commands are only available in mode OFFBOARD.
        self._change_param = rospy.ServiceProxy('/mavros/param/set', ParamSet)
        self.takeoffService = rospy.ServiceProxy('/mavros/cmd/takeoff', CommandTOL)
        
        #self.change_param_val(para="COM_POS_FS_DELAY",value=99)
        
        self.gazebo.pauseSim()

        rospy.logdebug("Finished MavROSEnv INIT...")


# Methods needed by the RobotGazeboEnv
    # ----------------------------
    
    def _reset_sim(self):
        """
        Including ***ekf2 stop*** and ***ekf2 start*** routines in original function
        """

        rospy.logdebug("RESET SIM START")
        if self.reset_controls :
            rospy.logdebug("RESET CONTROLLERS")
            self.gazebo.unpauseSim()
            self.controllers_object.reset_controllers()
            self._check_all_systems_ready()
            self._set_init_pose()
            rospy.logwarn("Set to initial pose")
            self.gazebo.pauseSim()
            self.gazebo.resetSim()
            self.gazebo.unpauseSim()
            
            rospy.logwarn("ekf2 module")
            ekf2_stop = subprocess.Popen([self.px4_ekf2_path, "stop"])
            ekf2_stop.wait()
            subprocess.Popen([self.px4_ekf2_path, "status"])
            rospy.sleep(2)
            rospy.logwarn("ekf2_stopped")
            ekf2_start = subprocess.Popen([self.px4_ekf2_path, "start"])
            ekf2_start.wait()
            subprocess.Popen([self.px4_ekf2_path, "status"])
            rospy.sleep(2)
            rospy.logwarn("ekf2_started")
            rospy.sleep(4)
            
            self.controllers_object.reset_controllers()
            self._check_all_systems_ready()
            rospy.logwarn("All systems checked")
            self.gazebo.pauseSim()
            
        else:
            self.gazebo.resetSim()
            rospy.logwarn("DONT RESET CONTROLLERS")
            self.gazebo.unpauseSim()
            self._check_all_systems_ready()
            self._set_init_pose()
            rospy.logwarn("Set to initial pose")
            self.gazebo.pauseSim()
            self.gazebo.resetSim()
            self.gazebo.unpauseSim()
            
            rospy.logwarn("ekf2 module")
            ekf2_stop = subprocess.Popen([self.px4_ekf2_path, "stop"])
            subprocess.Popen([self.px4_ekf2_path, "status"])
            rospy.logwarn("ekf2_stopped")
            rospy.sleep(2)
            ekf2_start = subprocess.Popen([self.px4_ekf2_path, "start"])
            subprocess.Popen([self.px4_ekf2_path, "status"])
            rospy.sleep(2)
            rospy.logwarn("ekf2_started")
            #rospy.sleep(4)

            self._check_all_systems_ready()
            rospy.logwarn("All systems checked")
            self.gazebo.pauseSim()
        rospy.logdebug("RESET SIM END")



    def _poseCb(self, msg):
        self._current_pose = msg

    def _stateCb(self, msg):
        self._current_state = msg
    
    def _gpsCb(self, msg):
        self._current_gps = msg
        
    

    def _check_all_systems_ready(self):
        """
        Checks that all the sensors, publishers, services and other simulation systems are
        operational.
        """
        self._check_all_sensors_ready()
        self._check_all_publishers_ready()
        return True
    

    def _check_all_sensors_ready(self):
        rospy.logdebug("CHECK ALL SENSORS CONNECTION:")
        self._check_current_pose_ready()
        self._check_current_state_ready()
        rospy.logdebug("All Sensors CONNECTED and READY!")
    
    def _check_current_pose_ready(self):
        self._current_pose = None
        rospy.logdebug("Waiting for /mavros/local_position/pose to be READY...")
        while self._current_pose is None and not rospy.is_shutdown():
            try:
                self._current_pose = rospy.wait_for_message("mavros/local_position/pose", PoseStamped, timeout=5.0)
                rospy.logdebug("Current mavros/local_position/pose READY=>")
            except:
                rospy.logdebug("Current mavros/local_position/pose not ready, retrying for getting lp_pose")
        return self._current_pose
    
    def _check_current_state_ready(self):
        rospy.logdebug("Waiting for /mavros/local_position/velocity_body to be READY...")
        while self._current_state is None and not rospy.is_shutdown():
            try:
                self._current_state = rospy.wait_for_message("mavros/state", State, timeout=5.0)
                rospy.logdebug("Current mavros/state READY=>")
            except:
                rospy.logerr("Current mavros/state not ready yet, retrying for getting current_state")
        return self._current_state


    def _check_all_publishers_ready(self):
        """
        Checks that all the publishers are working
        :return:
        """
        rospy.logdebug("CHECK ALL PUBLISHERS CONNECTION:")
        """ if (_control_mode == ControlMode.POSITION):
            self._check_local_pose_pub_connection()
        elif (_control_mode == ControlMode.VELOCITY):
            self._check_local_vel_pub_connection() """
        self._check_local_vel_pub_connection()
        rospy.logdebug("All Publishers CONNECTED and READY!")
        
    def _check_local_vel_pub_connection(self):

        while self._local_vel_pub.get_num_connections() == 0 and not rospy.is_shutdown():
            rospy.logdebug("Waiting for susbribers to _local_vel_pub...")
            try:
                self._rate.sleep()
            except rospy.ROSInterruptException:
                # This is to avoid error when world is rested, time when backwards.
                pass
        rospy.logdebug("_local_vel_pub Publisher Connected")
    
    def _check_local_pose_pub_connection(self):

        while self._local_pose_pub.get_num_connections() == 0 and not rospy.is_shutdown():
            rospy.logdebug("Waiting for susbribers to _local_pose_pub...")
            try:
                self._rate.sleep()
            except rospy.ROSInterruptException:
                # This is to avoid error when world is rested, time when backwards.
                pass
        rospy.logdebug("_local_pose_pub Publisher Connected")



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

    def ArmTakeOff(self, arm, alt = 5):
        req = CommandBoolRequest()
        d_req = CommandBoolRequest()
        if self._current_state.armed and arm:
            rospy.loginfo("already armed")
            
        else:
            # wait for service
            rospy.wait_for_service("mavros/cmd/arming")   
            # set request object
            req.value = arm
            d_req.value = not arm
             # zero time 
            t0 = rospy.get_time()
            # check response
            while not rospy.is_shutdown() and not self._current_state.armed:
                if rospy.get_time() - t0 > 2.0: # check every 5 seconds
                    try:
                        # request 
                        self._arming_client.call(req)
                    except rospy.ServiceException, e:
                        print ("Service did not process request")
                    t0 = rospy.get_time()
            rospy.logwarn("ARMED!!!")

        curr_alt = self._current_pose.pose.position.z
        try:
            ret = self.takeoffService(min_pitch=0, yaw=0, latitude=self._current_gps.latitude,\
                                longitude=self._current_gps.longitude, altitude=alt)
            if ret.success:
                rospy.logwarn("Took-off")
            else:
                rospy.loginfo("Failed taking-off")
        except rospy.ServiceException, e:
            print ("Service takeoff call failed")

        while not ret.success:
            rospy.logwarn("stuck in a loop!!!")
            self._rate.sleep()
            self._arming_client.call(req)
            ret = self.takeoffService(min_pitch=0, yaw=0, latitude=self._current_gps.latitude,\
                                longitude=self._current_gps.longitude, altitude=alt)
            if ret.success:
                rospy.loginfo("Took-off")
            else:
                rospy.loginfo("Failed taking-off")
    

    def ExecuteAction(self, vel_msg, epsilon=0.05, update_rate=20):

        rospy.logdebug("MavROS Base Twist Cmd>>" + str(vel_msg))
        self._check_local_vel_pub_connection()
        self._local_vel_pub.publish(vel_msg)

    def setMavMode(self, mode, timeout):
        """mode: PX4 mode string, timeout(int): seconds"""
        rospy.loginfo("setting FCU mode: {0}".format(mode))
        old_mode = self._current_state.mode
        loop_freq = 1  # Hz
        rate = rospy.Rate(loop_freq)
        mode_set = False
        for i in xrange(timeout * loop_freq):
            if self._current_state.mode == mode:
                mode_set = True
                rospy.loginfo("set mode success | seconds: {0} of {1}".format(i / loop_freq, timeout))
                break
            else:
                try:
                    res = self._set_mode_client(0, mode)  # 0 is custom mode
                    if not res.mode_sent:
                        rospy.logerr("failed to send mode command")
                except rospy.ServiceException as e:
                    rospy.logerr(e)
            rate.sleep()

    def LandDisArm(self):
        #Landing

        req = CommandBoolRequest()
        d_req = CommandBoolRequest()
        req.value = False
        d_req.value = True
        if self._current_pose.pose.position.z > 1:
            LandService = rospy.ServiceProxy('/mavros/cmd/land', CommandTOL)
            try:
                ret = LandService(min_pitch=0, yaw=0, latitude=self._current_gps.latitude,\
                                    longitude=self._current_gps.longitude, altitude=self._current_pose.pose.position.z)
                if ret.success:
                    rospy.loginfo("Took-off")
                else:
                    rospy.loginfo("Failed taking-off")
            except rospy.ServiceException, e:
                print ("Service takeoff call failed")

            while not ret.success:
                rospy.logwarn("stuck in a loop!!!")
                self._rate.sleep()
                self._arming_client.call(d_req)
                ret = LandService(min_pitch=0, yaw=0, latitude=self._current_gps.latitude,\
                                    longitude=self._current_gps.longitude, altitude=self._current_pose.pose.position.z)
                self._arming_client.call(req)
                if ret.success:
                    rospy.loginfo("Landing initiated")
                else:
                    rospy.loginfo("Failed to initiate landing")
        else:
            #Disarming
            if not self._current_state.armed:
                rospy.loginfo("already disarmed")
                
            else:
                # wait for service
                rospy.wait_for_service("mavros/cmd/arming")   
                # set request object
                # zero time 
                t0 = rospy.get_time()
                # check response
                while not rospy.is_shutdown() and self._current_state.armed:
                    if rospy.get_time() - t0 > 2.0: # check every 5 seconds
                        try:
                            # request 
                            self._arming_client.call(req)
                        except rospy.ServiceException, e:
                            print ("Service did not process request")
                        t0 = rospy.get_time()
                rospy.loginfo("DISARMED!!!")

    def get_current_pose(self):
        return self._current_pose

    def get_current_state(self):
        return self._current_state
    
    def get_current_gps(self):
        return self._current_gps


    def change_param_val(self, para="None", value=0, intflag=True):
        """
        Change parameter values through MavROS
        """
        rospy.wait_for_service('/mavros/param/set')
        try:
            param_val = ParamValue()
            if intflag:
                param_val.integer = value
            else:
                param_val.real = value
            ret = self._change_param(str(para), param_val)
            if ret.success:
                rospy.loginfo("Changed {0} to {1}".format(str(para),value))
            else:
                rospy.loginfo("Failed changing {0}".format(str(para)))
        except rospy.ServiceException, e:
            rospy.loginfo("Service call failed")