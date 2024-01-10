#!/usr/bin/env python
import numpy
import rospy
from openai_ros import robot_gazebo_env
from gazebo_msgs.msg import LinkStates

from std_msgs.msg import Float32MultiArray
from tf.transformations import euler_from_quaternion
from geometry_msgs.msg import Twist, Vector3
 

class StewartEnv(robot_gazebo_env.RobotGazeboEnv):
    """Stewart Platform Robot Environment.
    """
    def __init__(self):
        """Initializes a new Stewart environment.
        """
        print ("Entered Stewart Env")

        self.robot_name_space = "stewart"
        self.controllers_list = ['SDFJointController']

        print ("launch the init function of the Parent Class robot_gazebo_env_goal.RobotGazeboEnv.....")
        super(StewartEnv, self).__init__( controllers_list=self.controllers_list,
                                                robot_name_space=self.robot_name_space,
                                                reset_controls=False  # It is False for the stewart platform case, because of custom controller.
                                                )
        print ("launch the init function of the Parent Class robot_gazebo_env_goal.RobotGazeboEnv.....END")


        self.gazebo.unpauseSim()
        self._check_all_sensors_ready()
        print("check_all_sensors_ready checked.. ")

        # We Start all the ROS related Subscribers and publishers
        self.link_states_sub = rospy.Subscriber('gazebo/link_states', LinkStates, self.links_callback)


        # in the links_callback we publish the recived data info as below to use them later:
        self.pose_msg   = Twist()
        self.twist_msg  = Twist()
        self._pose_pub  = rospy.Publisher("/end_effector/pose" , Twist , queue_size=10)
        self._twist_pub = rospy.Publisher("/end_effector/twist", Twist , queue_size=10)


        # action publisher : if we provide required data to these publisher, we make the stewart move.
        self.init_joint_msg = Float32MultiArray()
        self.end_effector_pose_cmd = Twist()
        self.pid_values = Vector3()

        self._joints_cmd_pub   = rospy.Publisher('/stewart/legs_position_cmd',  Float32MultiArray, queue_size=10) # directly set the joints' length, here only for initialization.
        self._platform_cmd_pub = rospy.Publisher('/stewart/platform_pose',      Twist,             queue_size=10) # platform pose command to command wanted pose.
        self._pid_cmd_pub      = rospy.Publisher('/stewart/pid_cmd',            Vector3,           queue_size=10) # to change PID values

        self._check_publishers_connection()

        self.gazebo.pauseSim()                          

        print ("Entered Stewart Env END")


    # Methods needed by the RobotGazeboEnv:

    def _check_all_systems_ready(self):
        """
        Checks that all the sensors, publishers and other simulation systems are
        operational.
        """
        self._check_all_sensors_ready()
        return True


    def _check_all_sensors_ready(self):
        self._check_link_states_ready()
        
        rospy.logdebug("ALL gazebo feedbacks READY")
    
    def _check_link_states_ready(self):
        self.links = None
        while self.links is None and not rospy.is_shutdown():
            try:
                self.links = rospy.wait_for_message("/gazebo/link_states", LinkStates, timeout=1.0)
                rospy.logdebug("Current gazebo/link_states READY=>" + str(self.links))

            except:
                rospy.logerr("Current /gazebo/link_states not ready yet, retrying for getting gazebo/link_states")
        return self.links

    def _check_publishers_connection(self):
        """
        Checks that all the publishers are working
        :return:
        """
        rate = rospy.Rate(10)  # 10hz
        while self._platform_cmd_pub.get_num_connections() == 0 and not rospy.is_shutdown():
            rospy.logdebug("No susbribers to _platform_cmd_pub yet so we wait and try again")
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                # This is to avoid error when world is rested, time when backwards.
                pass
        rospy.logdebug("_platfrom pose Publisher Connected")


        while self._joints_cmd_pub.get_num_connections() == 0 and not rospy.is_shutdown():
            rospy.logdebug("No susbribers to ._joints_cmd_pub yet so we wait and try again")
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                # This is to avoid error when world is rested, time when backwards.
                pass
        rospy.logdebug("_joints_cmd_pub Publisher Connected")


        while self._pid_cmd_pub.get_num_connections() == 0 and not rospy.is_shutdown():
            rospy.logdebug("No susbribers to ._pid_cmd_pub yet so we wait and try again")
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                # This is to avoid error when world is rested, time when backwards.
                pass
        rospy.logdebug("_pid_cmd_pub Publisher Connected")

        rospy.logdebug("All Publishers READY")

  

    def links_callback(self, data):

        self.end_effector_pose= data.pose[2] # set the end effector link
        self.twist_msg = data.twist[2]
        
        orientation_list = [self.end_effector_pose.orientation.x,
                            self.end_effector_pose.orientation.y,
                            self.end_effector_pose.orientation.z, 
                            self.end_effector_pose.orientation.w
                            ]

        roll, pitch, yaw = euler_from_quaternion(orientation_list)

        self.pose_msg.linear.x  = self.end_effector_pose.position.x
        self.pose_msg.linear.y  = self.end_effector_pose.position.y
        self.pose_msg.linear.z  = self.end_effector_pose.position.z - 1.8625  # subtracting end_effector fram offset to get same measurement
        self.pose_msg.angular.x = roll
        self.pose_msg.angular.y = pitch
        self.pose_msg.angular.z = yaw


        ## Publish pose and twist
        self._pose_pub.publish(self.pose_msg)
        self._twist_pub.publish(self.twist_msg)

        
    def set_poistion_joints(self, initial_leg_pos):
        """
        Set initial position of the joints.
        """

        self.init_joint_msg.data = initial_leg_pos
      
        try:
            self._joints_cmd_pub.publish(self.init_joint_msg )
            result = True
        except Exception as ex:
            print(ex)
            result = False

        return result



    # ParticularEnv methods  ---- 
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


    # Methods that the TaskEnvironment will need.
    # ----------------------------
    def move_end_effector(self, x, y , z, roll, pitch , yaw):

        self.end_effector_pose_cmd.linear.x  = x
        self.end_effector_pose_cmd.linear.y  = y
        self.end_effector_pose_cmd.linear.z  = z
        self.end_effector_pose_cmd.angular.x = roll
        self.end_effector_pose_cmd.angular.y = pitch
        self.end_effector_pose_cmd.angular.z = yaw
        
        rospy.logdebug("Stewart [x, y, z , roll , pitch , roll] : >>" + str(self.end_effector_pose_cmd))

        self._platform_cmd_pub .publish(self.end_effector_pose_cmd)

        return self.end_effector_pose_cmd

    def set_pid_values(self, action_p, action_i , action_d):
        self.pid_values.x = action_p
        self.pid_values.y = action_i 
        self.pid_values.z = action_d

        rospy.loginfo(f"Pid Values: P: {round(action_p)}, I: {round(action_i)}, D: {round(action_d)}")
        try:
            self._pid_cmd_pub.publish(self.pid_values)
        except Exception as ex:
            print(ex)

    def get_end_effector_pose(self):
        return self.pose_msg 

    def get_end_effector_twist(self):
        return self.twist_msg