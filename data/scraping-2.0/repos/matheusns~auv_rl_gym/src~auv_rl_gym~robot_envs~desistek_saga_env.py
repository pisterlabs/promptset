import rospy
import time

# msgs
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist 

from openai_ros import robot_gazebo_env

class DesistekSagaEnv(robot_gazebo_env.RobotGazeboEnv):
    def __init__(self):
        rospy.logdebug("DesistekSagaEnv: Starting environment")
        
        self.init_attributes()
        self.init_ROS_attributes()

        super(DesistekSagaEnv, self).__init__(controllers_list=self.controllers_list,
                                            robot_name_space=self.robot_name_space,
                                            reset_controls=False,
                                            start_init_physics_parameters=False,
                                            reset_world_or_sim="NO_RESET_SIM")
        self.wait_simulation()
        rospy.loginfo("DesistekSagaEnv: Finished DesistekSagaEnv initialization")

    def init_attributes(self):
        self.controllers_list = []
        self.robot_name_space = "" 
        self.publishers_array = []
        
    def init_ROS_attributes(self):
        if not rospy.has_param('/odom_topic') or not rospy.has_param('/velocity_topic'):
            rospy.logerr("Required parameters not set! Please ensure both velocity_topic and odom_topic are set.")
            return

        self.odom_topic_name = rospy.get_param('/odom_topic', '/odom_topic')
        rospy.Subscriber(self.odom_topic_name, Odometry, self._odom_callback)

        self.velocity_topic_name = rospy.get_param('/velocity_topic', '/velocity_topic')
        self._cmd_drive_pub = rospy.Publisher(self.velocity_topic_name, Twist, queue_size=1)
        self.publishers_array.append(self._cmd_drive_pub)
        
    def _odom_callback(self, data):
        self.odom = data
        
    def _get_odom(self):
        return self.odom

    def wait_simulation(self):
        self.gazebo.unpauseSim()
        self._check_all_systems_ready()
        self.gazebo.pauseSim()
        
    def _check_all_systems_ready(self):
        rospy.logdebug("DesistekSagaEnv: checking all systems...")
        self._check_all_sensors_ready()
        self._check_all_publishers_ready()
        rospy.loginfo("DesistekSagaEnv: system is ready")
        return True

    def _check_all_sensors_ready(self):
        self._check_odom_ready()

    def _check_odom_ready(self):
        self.odom = None
        while self.odom is None and not rospy.is_shutdown():
            try:
                self.odom = rospy.wait_for_message(self.odom_topic_name, Odometry, timeout=1.0)
            except:
                rospy.logerr("DesistekSagaEnv: current %s not ready yet", self.odom_topic_name)

        rospy.loginfo("DesistekSagaEnv: current %s is ready", self.odom_topic_name)
    
    def _check_all_publishers_ready(self):
        for publisher_object in self.publishers_array:
            self._check_pub_connection(publisher_object)
        rospy.logdebug("DesistekSagaEnv: all Publishers READY")

    def _check_pub_connection(self, publisher_object):
        rate = rospy.Rate(10)  # 10hz
        while publisher_object.get_num_connections() == 0 and not rospy.is_shutdown():
            rospy.logdebug("DesistekSagaEnv: no susbribers to publisher_object yet so we wait and try again")
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                pass
    
    # Methods that the TrainingEnvironment will need to define here as virtual
    # because they will be used in RobotGazeboEnv GrandParentClass and defined in the
    # TrainingEnvironment.
    # ----------------------------
    def _set_action(self, action):
        """Applies the given action to the simulation.
        """
        raise NotImplementedError()

    def _set_init_pose(self):
        """Sets the Robot in its init pose
        """
        raise NotImplementedError()
    
    def _init_env_variables(self):
        """Inits variables needed to be initialised each time we reset at the start
        of an episode.
        """
        raise NotImplementedError()

    def _compute_reward(self, observations, terminated):
        """Calculates the reward to give based on the observations given.
        """
        raise NotImplementedError()

    def _get_obs(self):
        raise NotImplementedError()

    def _is_terminated(self, observations):
        """Checks if episode done based on observations given.
        """
        raise NotImplementedError()

    def _is_truncated(self, observations):
        """Checks if episode truncation conditions are satisfied.
        """
        raise NotImplementedError()
        
    # Methods that the TrainingEnvironment will need.
    # ----------------------------
    def wait_time_for_execute_movement(self, time_sleep):
        time.sleep(time_sleep)
    


