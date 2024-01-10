import os
import numpy as np
import copy
import rospy
import time
from gym import spaces
from pyquaternion import Quaternion
from openai_ros.robot_envs import modrob_env_moveit
from gym.envs.registration import register
from geometry_msgs.msg import Point
from geometry_msgs.msg import Pose
from std_msgs.msg import Float32
from gazebo_msgs.msg import ModelStates, ModelState
from gazebo_msgs.srv import SetModelState
from openai_ros.task_envs.task_commons import LoadYamlFileParamsTest
from openai_ros.openai_ros_common import ROSLauncher
from modrob_simulation.msg import Collision, Collisions

class ModRobSphereEnvMoveIt(modrob_env_moveit.ModRobEnvMoveIt):
    def __init__(self):
        """
        This Task Env is designed for having the ModRob with an example moving obstacle.
        It will learn how to finish tasks without colliding with the obstacle.
        """
        # The robot name parameter must be in the parameter server
        self.robot_name_ = rospy.get_param('/modrob/robot_name')    
        # This is the path where the simulation files, the Task and the Robot gits will be downloaded if not there
        ros_abs_env_var = rospy.get_param("/modrob/ros_abs_env_var", None)
        try:  
            ros_ws_abspath = os.environ[ros_abs_env_var]
        except: 
            print("Please set the environment variable {}".format(ros_abs_env_var))
            sys.exit(1)
        assert ros_ws_abspath is not None, "You forgot to set ros_ws_abspath in your yaml file of your main RL script. Set ros_ws_abspath: \'YOUR/SIM_WS/PATH\'"
        assert os.path.exists(ros_ws_abspath), "The Simulation ROS Workspace path " + ros_ws_abspath + \
                                               " DOESNT exist, execute: mkdir -p " + ros_ws_abspath + \
                                               "/src;cd " + ros_ws_abspath + ";catkin_make"

        ROSLauncher(rospackage_name="modrob_simulation",
                    launch_file_name="start_world_sphere.launch",
                    ros_ws_abspath=ros_ws_abspath)

        # Load Params from the desired Yaml file
        LoadYamlFileParamsTest(rospackage_name="openai_ros",
                               rel_path_from_package_to_file="src/openai_ros/task_envs/human_modrob/config",
                               yaml_file_name="modrob_human_moveit.yaml")

        # Here we will add any init functions prior to starting the MyRobotEnv
        super(ModRobSphereEnvMoveIt, self).__init__(ros_ws_abspath)
        
        ## Load in environment variables
        self._get_env_variables()
        # We set the reward range, which is not compulsory but here we do it.
        self.reward_range = (-np.inf, np.inf)

        ## Set action and observation space
        # Continuous action space.
        # All actions should range from 0 to 1. This improves training.
        # Right now, only arm action and observation is implemented.
        # TODO: Add gripper support.
        self.n_actions = len(self.get_arm_joint_names())
        # number of arm joints + 
        # current end effector position (x,y,z) + 
        # current end effector orientation quaternion (x, y, z, w) +
        # goal end effector position (x,y,z) + 
        # goal end effector orientation quaternion (x, y, z, w) + 
        # sphere obstacle position (x, y, z) 
        self.n_observations = len(self.get_arm_joint_names()) + 3 + 4 + 3 + 4 + 3
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.n_actions,))
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.n_observations,))
        rospy.logdebug("ACTION SPACES TYPE===>"+str(self.action_space))
        rospy.logdebug("OBSERVATION SPACES TYPE===>"+str(self.observation_space))

        self.cumulated_steps = 0.0

        # Set starting position of spherical obstacle.
        # This will be replaced by a more sophisticated obstacle in the future.
        self.sphere_obs_start_pose = self.create_pose([0, 2, 0.7], [1, 0, 0, 0])
        self.sphere_start_velo = -0.5

        ## Unpause sim and start all subscribers and publishers
        self.gazebo.unpauseSim()
        # Collision detection topic
        self._collision_topic = '/' + self.robot_name_ + '/collisions'
        # ROS subscriber to robot collision
        rospy.Subscriber(self._collision_topic, Collisions, self._collision_callback)
        self._sphere_sensor_topic = '/sphere_obs/position'
        # ROS subscriber to sphere position
        rospy.Subscriber(self._sphere_sensor_topic, Point, self._sphere_pos_callback)
        self._check_sphere_pos_ready()
        self._sphere_pose_topic = '/sphere_obs/pose_cmd'
        self._sphere_pose_publisher = rospy.Publisher(self._sphere_pose_topic, Pose, queue_size=20)
        self._sphere_vel_topic = '/sphere_obs/vel_cmd'
        self._sphere_vel_publisher = rospy.Publisher(self._sphere_vel_topic, Float32, queue_size=10)
        ## Utility for moving start and goal position visual models
        # CAREFUL: The getter method has an s at the end while the setter method doesn't!
        # Ros subscriber to gazebo model state
        self._gazebo_model_state_topic = '/gazebo/model_states'
        rospy.Subscriber(self._gazebo_model_state_topic, ModelStates, self._gazebo_model_state_callback)
        self._check_gazebo_model_state_ready()
        # Get the base pose
        self.base_pose = self._get_base_pose()
        self.gazebo.pauseSim()

    def _get_env_variables(self):
        """Load in environment variables from yaml.
        Relevant variables:
            joint_min: Minimal angle for all joints
            joint_max: Maximal angle for all joints
            joint_max_delta: Max theoretically allowed movement per execution step (in rad)
            gripper1_min: Minimal position of gripper part 1
            gripper1_max: Maximal position of gripper part 1
            gripper2_min: Minimal position of gripper part 2
            gripper2_max: Maximal position of gripper part 2
            gripper_max_delta: Maximal theoretically allowed movement per execution step (in m)
            ee_limits: Overapproximative end effector position limits
                .max.x, .y, .z
                .min.x, .y, .z
            base_pose: geometry_msgs.msg.Pose for base
            use_delta_actions: True: Use differential position commands, False: Use absolute position commands
            movement_error: Precision maximum for regular movement (can be arbitrarely small)
            movement_timeout: Time waited until next movement execution
            init_error: Precision for start position
            goal_error_position: 0.2 # Precision for goal reached euclidean distance
            goal_error_orientation: 0.2 # Precision for goal reached quaternion distance
            init_joint0_position: Initial position for joint0 - TODO: Change this to dynamically.
            init_joint1_position: Initial position for joint1 - TODO: Change this to dynamically.
            init_joint2_position: Initial position for joint2 - TODO: Change this to dynamically.
            init_hand_to_finger1_position: Initial position for gripper part 1 - TODO: Change this to dynamically.
            init_hand_to_finger2_position: Initial position for gripper part 2 - TODO: Change this to dynamically.
            goal_pose: geometry_msgs.msg.Pose for end effector goal
            distance_penalty_position: Reward penalty for position distance
            distance_penalty_orientation: Reward penalty for orientation distance
            time_penalty: Time penalty for every step
            goal_reward: Points given when reaching the goal
            collision_penalty: Penalty when colliding with an object
        """
        ## Determine the normalization constants for all observations and actions.
        # action_norm consists of two columns (c, r) and one row per controllable joint.
        # To normalize an action a, calculate: a_n = (a-c)/r, r!=0
        # To denormalize an normalized action a_n, calculate a = (a_n * r) + c
        # Since we are using the tanh as activation for actions, we normalize a to [-1; 1]
        self.action_norm = []
        self.observation_norm = []
        # Max movements
        self.joint_max_delta = rospy.get_param('/modrob/joint_max_delta')
        self.gripper_max_delta = rospy.get_param('/modrob/gripper_max_delta')
        self.use_delta_actions = rospy.get_param('/modrob/use_delta_actions')
        # TODO: Read these from urdf file.
        self.joint_min = rospy.get_param('/modrob/joint_min')
        self.joint_max = rospy.get_param('/modrob/joint_max')
        assert self.joint_max-self.joint_min != 0, "Joint difference is zero"
        self.gripper1_min = rospy.get_param('/modrob/gripper1_min')
        self.gripper1_max = rospy.get_param('/modrob/gripper1_max')
        assert self.gripper1_max-self.gripper1_min != 0, "Gripper 1 difference is zero"
        self.gripper2_min = rospy.get_param('/modrob/gripper2_min')
        self.gripper2_max = rospy.get_param('/modrob/gripper2_max')
        assert self.gripper2_max-self.gripper2_min != 0, "Gripper 2 difference is zero"
        # First entries are joint positions
        self.observation_id_joints = 0 # defines where the joint values start in observation
        for joint_name in self.get_arm_joint_names():
            if self.use_delta_actions:
                _c = -1*self.joint_max_delta
                _r = 2*self.joint_max_delta    
            else:
                _c = self.joint_min
                _r = self.joint_max-self.joint_min
            # From [0; 1] normalization to [-1; 1]
            c = _c + _r/2
            r = _r/2
            self.action_norm.append([c, r])
            self.observation_norm.append([self.joint_min, self.joint_max-self.joint_min])
            """ Gripper normalization. Add back to code in the future.
            elif "hand_to_finger1" in joint_name:
                # Only one action for both grippers (they move together)
                if self.use_delta_actions:
                    self.action_norm.append([-1*self.gripper_max_delta, 2*self.gripper_max_delta])
                else:
                    self.action_norm.append([self.gripper1_min, self.gripper1_max-self.gripper1_min])
                self.observation_norm.append([self.gripper1_min, self.gripper1_max-self.gripper1_min])
            """
        # Add normalization for current and goal ee position
        self.observation_id_current_ee_pose = len(self.observation_norm) # defines where the current ee pose values start in observation
        self.observation_id_goal_ee_pose = len(self.observation_norm) + 7 # defines where the goal ee pose values start in observation
        for i in range(2):
            ee_limits = rospy.get_param('/modrob/ee_limits')
            self.observation_norm.append([ee_limits.get("min").get("x"), ee_limits.get("max").get("x")-ee_limits.get("min").get("x")])
            self.observation_norm.append([ee_limits.get("min").get("y"), ee_limits.get("max").get("y")-ee_limits.get("min").get("y")])
            self.observation_norm.append([ee_limits.get("min").get("z"), ee_limits.get("max").get("z")-ee_limits.get("min").get("z")])
            # Add normalization for ee quaternion orientation
            for _ in range(4):
                self.observation_norm.append([0, 1])
        # Add normalization for sphere obstacle. 
        # This will always be a bit hacky since there is no definite max and min position for any obstacle.
        self.observation_norm.append([-3, 3]) #x
        self.observation_norm.append([-3, 3]) #y
        self.observation_norm.append([0, 3]) #z
        self.action_norm = np.array(self.action_norm)
        self.observation_norm = np.array(self.observation_norm)
        # Movement settings
        self.movement_error = rospy.get_param('/modrob/movement_error')
        self.movement_timeout = rospy.get_param('/modrob/movement_timeout')
        self.init_error = rospy.get_param('/modrob/init_error')
        self.goal_error_position = rospy.get_param('/modrob/goal_error_position')
        self.goal_error_orientation = rospy.get_param('/modrob/goal_error_orientation')
        # Set initial joint positions
        # Right now, only arm position movement implemented!
        # TODO: Add init gripper position
        self.init_arm_joint_position = []
        if rospy.has_param("/modrob/init_joint_position"):
            self.init_arm_joint_position = rospy.get_param("/modrob/init_joint_position")
        assert(len(self.init_arm_joint_position) == len(self._arm_joint_names))
        # Goal and reward
        goal_pose_position = rospy.get_param('/modrob/goal_pose').get("position")
        goal_pose_orientation = rospy.get_param('/modrob/goal_pose').get("orientation")
        self.goal_pose = self.create_pose([goal_pose_position.get("x"), goal_pose_position.get("y"), goal_pose_position.get("z")], 
                                           [goal_pose_orientation.get("x"), goal_pose_orientation.get("y"), goal_pose_orientation.get("z"), goal_pose_orientation.get("w")])
        self.goal_position = [self.goal_pose.position.x, 
                              self.goal_pose.position.y, 
                              self.goal_pose.position.z]
        self.goal_quaternion = [self.goal_pose.orientation.x,
                                self.goal_pose.orientation.y,
                                self.goal_pose.orientation.z,
                                self.goal_pose.orientation.w]
        self.goal_observation = self.goal_position + self.goal_quaternion
        self.distance_penalty_position = rospy.get_param('/modrob/distance_penalty_position')
        self.distance_penalty_orientation = rospy.get_param('/modrob/distance_penalty_orientation')
        self.time_penalty = rospy.get_param('/modrob/time_penalty')
        self.goal_reward = rospy.get_param('/modrob/goal_reward')
        self.collision_penalty = rospy.get_param('/modrob/collision_penalty')


    def _set_base_pose(self):
        """Set the pose of the robots base."""
        assert bool(self.robot_name_ in self.gazebo_model_pose), self.robot_name_ + " not in gazebo model states!"
        self.move_gazebo_model(self.robot_name_, self.base_pose)

    def _get_base_pose(self):
        """Set the pose of the robots base."""
        assert bool(self.robot_name_ in self.gazebo_model_pose), self.robot_name_ + " not in gazebo model states!"
        return self.gazebo_model_pose[self.robot_name_]

    def _set_init_pose(self):
        """Sets the Robot in its init pose.
        """
        # Move until init position is reached (timeout=0)
        self.move_arm_joints(self.init_arm_joint_position, error=self.init_error, timeout=0.0)     
        self.init_pose = self.get_ee_pose()
        return True


    def _init_env_variables(self):
        """Inits episode specific variables each time we reset at the start of an episode.
        """
        # For Info Purposes
        self.cumulated_reward = 0.0
        # Set to false Done, because its calculated asyncronously
        self._episode_done = False
        self.last_arm_joint_position = self.get_arm_joint_positions()
        self.is_collided = False
        # Place start and goal markers
        self.move_gazebo_model("start_pos", self.correct_model_pose(self.init_pose))
        self.move_gazebo_model("goal_pos", self.correct_model_pose(self.goal_pose))
        # Place sphere obstacle
        self._sphere_pose_publisher.publish(self.sphere_obs_start_pose)
        self._sphere_vel_publisher.publish(self.sphere_start_velo)


    def _set_action(self, action):
        """Give a control command to the robot.
        First, the action is clipped to the action space.
        It is possible to assign negative rewards for too high actions.
        This function denormalizes the action command and controls the robot.
        Args: 
            action (array): Normalized actions
        """
        action = np.clip(action, self.action_space.low, self.action_space.high)
        rospy.logdebug("Start Set Action ==>"+str(action))
        # Denormalize actions
        denormalized_action = self.denormalize_actions(action)
        # Build joint position dict
        if self.use_delta_actions:
            joint_positions = self.create_joint_positions_delta(denormalized_action)
        else:
            joint_positions = self.create_joint_positions_absolute(denormalized_action)
        # Set action as command
        # Only arm movement implemented right now. TODO: Add gripper action.
        self.move_arm_joints(joint_positions, error=self.movement_error, timeout=self.movement_timeout)
        rospy.logdebug("END Set Action ==>"+str(action))


    def _get_obs(self):
        """Get normalized observation array from robot sensors.
        Returns:
            observations (array): Normalized observation array 
        """
        rospy.logdebug("Start Get Observation ==>")
        # Get non-normalized observations
        observations = self.retrieve_observations()
        # Normalize observations
        observations = self.normalize_observations(observations)
        rospy.logdebug("END Get Observation ==>"+str(observations))
        return observations


    def _is_done(self, observations):
        """Return if episode is finished."""
        if self.is_collided:
            return True
        observations = self.denormalize_observations(observations)
        current_ee_position = observations[self.observation_id_current_ee_pose:self.observation_id_current_ee_pose+3]
        current_ee_quaternion = observations[self.observation_id_current_ee_pose+3:self.observation_id_current_ee_pose+7]
        current_ee_pose = self.create_pose(current_ee_position, current_ee_quaternion)
        return self.is_in_goal_pose(current_ee_pose=current_ee_pose, 
                                    epsilon_position=self.goal_error_position, 
                                    epsilon_orientation=self.goal_error_orientation)

    def _compute_reward(self, observations, done):
        """Compute reward for this step."""
        reward = 0
        # We run this twice, once in _is_done and once here. Check whether this is computational heavy and maybe safe results.
        observations = self.denormalize_observations(observations)
        current_ee_position = observations[self.observation_id_current_ee_pose:self.observation_id_current_ee_pose+3]
        current_ee_quaternion = observations[self.observation_id_current_ee_pose+3:self.observation_id_current_ee_pose+7]
        current_ee_pose = self.create_pose(current_ee_position, current_ee_quaternion)
        position_distance, orientation_distance = self.get_distances_from_desired_pose(current_ee_pose)
        if not done:
            # Penalty per time step
            reward -= self.time_penalty
            # Penalty for distance from goal position
            reward -= self.distance_penalty_position * position_distance
            # Penalty for distance from goal orientation
            reward -= self.distance_penalty_orientation * orientation_distance
        else:
            # The done flag is set either when the goal is reached or a collision occured.
            if self.is_collided:
                reward -= self.collision_penalty
            else:
                reward += self.goal_reward

        rospy.logdebug("This step reward = " + str(reward))
        self.cumulated_reward += reward
        rospy.logdebug("Cumulated_reward = " + str(self.cumulated_reward))
        self.cumulated_steps += 1
        rospy.logdebug("Cumulated_steps = " + str(self.cumulated_steps))
        return reward

    def _collision_callback(self, data):
        """This function is called if a collision is detected by one of the modrobs collision sensors.
        The collision sensor plugins publish the collision information on /self.robot_name_/collisions.
        Sets self.is_collided to true.
        Outputs an info message.
        """
        if not self.is_collided:
            self.is_collided = True
            rospy.logwarn("Collision detected between {} and {}.".format(
                data.collisions[0].parent_contact, data.collisions[0].obstacle_contact
            ))
    
    def _sphere_pos_callback(self, data):
        """Get the estimated position of the sphere (a Kalman filter is already applied)."""
        self.sphere_pos = [data.x, data.y, data.z]


    def create_joint_positions_absolute(self, actions):
        """Creates joint_positions from an absolute action array.
        Args:
            actions: Action array (This should be denormalized!), shape = [n_actions]
        Returns:
            joint_positions (list): desired joint positions.
        """
        joint_positions = np.clip(actions, self.joint_min, self.joint_max)
        """ Code for gripper action. Include in code in the future.
            elif "hand_to_finger" in action_name:
                joint_positions[action_name] = np.clip(actions[i], self.gripper1_min, self.gripper1_max)
                # Set action for finger 2 reversed to finger 1.
                # F1 = min --> F2 = max; F1 = max --> F2 = max
                # F2 = ax + b
                a = (self.gripper2_min - self.gripper2_max)/(self.gripper1_max - self.gripper1_min)
                b = (self.gripper1_max*self.gripper2_max-self.gripper1_min*self.gripper2_min)/(self.gripper1_max - self.gripper1_min)
                joint_positions["hand_to_finger2"] = a * actions[i] + b
        """
        return joint_positions

    def create_joint_positions_delta(self, actions):
        """Creates absolute joint_positions from an delta action array.
        Args:
            actions: Action array (This should be denormalized!), shape = [n_actions]
        Returns:
            joint_positions (list): desired absolute joint position.
        """
        # Use the last observed joint position (not the current!)
        last_arm_joint_positions = self.last_arm_joint_position
        joint_positions = np.clip(last_arm_joint_positions + actions, self.joint_min, self.joint_max)
        """ Code for gripper action. Include in code in the future.
        elif "hand_to_finger" in joint_name:
            joint_positions[joint_name] = np.clip(last_joint_positions[joint_name] + actions[i], self.gripper1_min, self.gripper1_max)
            # Set action for finger 2 reversed to finger 1.
            # F1 = min --> F2 = max; F1 = max --> F2 = max
            # F2 = ax + b
            a = (self.gripper2_min - self.gripper2_max)/(self.gripper1_max - self.gripper1_min)
            b = (self.gripper1_max*self.gripper2_max-self.gripper1_min*self.gripper2_min)/(self.gripper1_max - self.gripper1_min)
            joint_positions["hand_to_finger2"] = a * joint_positions[joint_name] + b
        """
        return joint_positions

    def retrieve_observations(self):
        """Retrieve all observations (not normalized).
        Sets the last observed joint position.
        Observation consists of:
            - arm_joint_positions
            - current ee position (cartesian) and orientation (quaternion)
            - goal ee position (cartesian) and orientation (quaternion)
            - sphere obstacle position (cartesian)
        Returns:
            observations (list): non normalized observations, shape = [n_observations]
        """
        self.last_arm_joint_position = self.get_arm_joint_positions()
        observations = self.last_arm_joint_position + self.get_ee_position() + self.get_ee_quaternion() + self.goal_observation + self.sphere_pos
        return observations

    # Internal TaskEnv Methods

    def normalize_actions(self, actions):
        """Normalize an array of actions.
        To normalize an action a, calculate: a_n = (a-c)/r, r!=0
        Args:
            actions: Action array, shape = [n_actions]
        Returns:
            normalized_actions: Normalized action array, shape = [n_actions]
        """
        normalized_actions = []
        if len(actions) == 0:
            rospy.logerr("No actions to normalize.")
            return normalized_actions
        normalized_actions = (actions - self.action_norm[:, 0]) / self.action_norm[:, 1]
        return normalized_actions

    def denormalize_actions(self, normalized_actions):
        """Denormalize an array of actions.
        To denormalize an normalized action a_n, calculate a = (a_n * r) + c
        Args:
            normalized_actions: Normalized action array, shape = [n_actions]
        Returns:
            actions: Action array, shape = [n_actions]
        """
        actions = []
        if len(normalized_actions) == 0:
            rospy.logerr("No actions to denormalize.")
            return actions
        actions = (normalized_actions * self.action_norm[:, 1]) + self.action_norm[:, 0]
        return actions

    def normalize_observations(self, observations):
        """Normalize an array of observations.
        To normalize an observation a, calculate: a_n = (a-c)/r, r!=0
        Args:
            observations: Action array, shape = [n_observations]
        Returns:
            normalized_observations: Normalized observation array, shape = [n_observations]
        """
        normalized_observations = []
        if len(observations) == 0:
            rospy.logwarn("No observations to normalize.")
            return normalized_observations
        normalized_observations = (observations - self.observation_norm[:, 0]) / self.observation_norm[:, 1]
        return normalized_observations

    def denormalize_observations(self, normalized_observations):
        """Denormalize an array of observations.
        To denormalize an normalized observation a_n, calculate a = (a_n * r) + c
        Args:
            normalized_observations: Normalized observation array, shape = [n_observations]
        Returns:
            observations: Action array, shape = [n_observations]
        """
        observations = []
        if len(normalized_observations) == 0:
            rospy.logwarn("No observations to denormalize.")
            return observations
        observations = (normalized_observations * self.observation_norm[:, 1]) + self.observation_norm[:, 0]
        return observations

    def discretize_observation(self,data,new_ranges):
        """
        Discards all the laser readings that are not multiple in index of new_ranges
        value.
        """
        self._episode_done = False

        discretized_ranges = []
        mod = len(data.ranges)/new_ranges

        rospy.logdebug("data=" + str(data))
        rospy.logwarn("new_ranges=" + str(new_ranges))
        rospy.logwarn("mod=" + str(mod))

        for i, item in enumerate(data.ranges):
            if (i%mod==0):
                if item == float ('Inf') or np.isinf(item):
                    discretized_ranges.append(self.max_laser_value)
                elif np.isnan(item):
                    discretized_ranges.append(self.min_laser_value)
                else:
                    discretized_ranges.append(int(item))

                if (self.min_range > item > 0):
                    rospy.logerr("done Validation >>> item=" + str(item)+"< "+str(self.min_range))
                    self._episode_done = True
                else:
                    rospy.logwarn("NOT done Validation >>> item=" + str(item)+"< "+str(self.min_range))


        return discretized_ranges


    def is_in_goal_pose(self, current_ee_pose, epsilon_position=0.05, epsilon_orientation=0.05):
        """Checks whether the end effector is within a margin of error to its goal pose.
        Args:
            current_ee_pose (geometry_msgs.msg.Pose): current pose of the end effector
            epsilon_position (double): margin of error for position (euclidean distance)
            epsilon_orientation (double): margin of error for orientation  
        """
        assert isinstance(current_ee_pose, Pose), "current_ee_pose is not of type geometry_msgs.msg.Pose!"
        # Calcualte distances
        position_distance, orientation_distance = self.get_distances_from_desired_pose(current_ee_pose)
        return position_distance <= epsilon_position and orientation_distance <= epsilon_orientation


    def get_distances_from_desired_pose(self, current_ee_pose):
        """Calculates the euclidean distance and orientation distance from the current ee pose to the goal pose.
        Args:  
            current_ee_pose (geometry_msgs.msg.Pose): current pose of the end effector
        Returns:
            position_distance (double): euclidean distance between cartesian ee positions
            orientation_distance (double): quaternion distance between the ee quaternions
        """
        assert isinstance(current_ee_pose, Pose), "current_ee_pose is not of type geometry_msgs.msg.Pose!"
        position_distance = self.calculate_ee_position_distance(current_ee_pose)
        orientation_distance = self.calculate_gripper_orientation_distance(current_ee_pose)
        return position_distance, orientation_distance

    def calculate_ee_position_distance(self, current_ee_pose):
        """Calculate euclidean distance between the current and goal end effector position (goal in self.).
        Args:
            current_ee_pose (geometry_msgs.msg.Pose): Current end effector pose
        Returns:
            Euclidean distance between the two poses.
        """
        assert isinstance(current_ee_pose, Pose), "current_ee_pose is not of type geometry_msgs.msg.Pose!"
        c_ee_pos = current_ee_pose.position
        g_ee_pos = self.goal_pose.position
        return np.sqrt((c_ee_pos.x - g_ee_pos.x)**2 + 
                       (c_ee_pos.y - g_ee_pos.y)**2 + 
                       (c_ee_pos.z - g_ee_pos.z)**2)

    def calculate_ee_orientation_distance(self, current_ee_pose):
        """Calculate distance between the current and goal end effector pose quaternion (goal in self.).
        We use http://kieranwynn.github.io/pyquaternion/#distance-computation for the quaternion distance calculation.
        Note: This function does not measure the distance on the hypersphere, but it takes into account the fact that q and -q encode the same rotation. 
        It is thus a good indicator for rotation similarities.
        Args:
            current_ee_pose (geometry_msgs.msg.Pose): Current end effector pose
        Returns:
            Distance between the two quaternions.
        """
        assert isinstance(current_ee_pose, Pose), "current_ee_pose is not of type geometry_msgs.msg.Pose!"
        q0 = self.convert_pose_to_quaternion(current_ee_pose)
        q1 = self.convert_pose_to_quaternion(self.goal_pose)
        return Quaternion.absolute_distance(q0, q1)

    def calculate_gripper_orientation_distance(self, current_ee_pose):
        """Calculate distance between the current and goal end effector z-axis.
        Returns the great circle distance between the z-axis of the end effector and the goal pose.
        The rotation around the z-axis is assumed to be irrelevant for gripping objects for this function.
        Args:
            current_ee_pose (geometry_msgs.msg.Pose): Current end effector pose
        Returns:
            Distance between the two quaternions.
        """
        assert isinstance(current_ee_pose, Pose), "current_ee_pose is not of type geometry_msgs.msg.Pose!"
        q0 = self.convert_pose_to_quaternion(current_ee_pose)
        z0 = self.get_new_z_axis(q0)
        q1 = self.convert_pose_to_quaternion(self.goal_pose)
        z1 = self.get_new_z_axis(q1)
        return self.great_circle_distance(z0, z1)
    
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

    def correct_model_pose(self, pose):
        """Correct the model pose by the pose of the base.
        This is needed because the ee poses of moveit are in relation to the base.
        TODO: Include orientation!
        Args:
            pose (geometry_msgs.msg.Pose)
        Returns:
            corrected copy of pose
        """
        new_pose = copy.deepcopy(pose)
        new_pose.position.x += self.base_pose.position.x
        new_pose.position.y += self.base_pose.position.y
        new_pose.position.z += self.base_pose.position.z
        return new_pose

    def get_model_pose(self, model_name):
        """Return the pose of a gazebo model by name.
        Args:
            model_name (String): Name of the model (in world file)
        Returns:
            pose (geometry_msgs.msg.Pose)
        """
        if model_name in self.gazebo_model_pose:
            return self.gazebo_model_pose[model_name]
        else:
            rospy.logerr("Model {} does not exist in gazebo world.".format(model_name))
            return None


    def move_gazebo_model(self, model_name, pose):
        """ Move the gazebo model to the desired pose
        Args:
            model_name (string): name of the model (Must be in topic /gazebo/model_states)
            pose (geometry_msgs.msg.Pose)
        """
        if model_name in self.gazebo_model_pose:
            state_msg = SetModelState()
            state_msg.model_name = model_name
            state_msg.pose = pose
            state_msg.twist = self.gazebo_model_twist[model_name]
            state_msg.reference_frame = "world"
            result = self.publish_gazebo_model_state(state_msg)
        else:
            result = False
            rospy.logwarn("The goal_pos model does not exist!")
        return result


    def publish_gazebo_model_state(self, model_state):
        """Publish a gazebo model state.
        Args:
            model_state (gazebo_msgs.srv.SetModelState)
        """
        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            resp = set_state(model_state = model_state)
        except rospy.ServiceException as e:
            rospy.logerr("Set model state service call failed: %s"%e)
            resp = False
        return resp

    def _check_gazebo_model_state_ready(self):
        self.gazebo_model_pose = dict()
        self.gazebo_model_twist = dict()
        rospy.logdebug("Waiting for {} to be READY...".format(self._gazebo_model_state_topic))
        while not self.gazebo_model_pose and not rospy.is_shutdown():
            try:
                data = rospy.wait_for_message(self._gazebo_model_state_topic, ModelStates, timeout=5.0)
                self.gazebo_model_pose = dict(zip(data.name, data.pose))
                self.gazebo_model_twist = dict(zip(data.name, data.twist))
                rospy.logdebug("Current {} READY=>".format(self._gazebo_model_state_topic))

            except:
                rospy.logerr("Current {} not ready yet, retrying for getting gazebo_model states".format(self._gazebo_model_state_topic))

        return self.gazebo_model_pose
        
    def _check_sphere_pos_ready(self):
        self.sphere_pos = None
        rospy.logdebug("Waiting for {} to be READY...".format(self._sphere_sensor_topic))
        while not self.sphere_pos and not rospy.is_shutdown():
            try:
                data = rospy.wait_for_message(self._sphere_sensor_topic, Point, timeout=5.0)
                self.sphere_pos = [data.x, data.y, data.z]
                rospy.logdebug("Current {} READY=>".format(self._sphere_sensor_topic))

            except:
                rospy.logerr("Current {} not ready yet, retrying for getting gazebo_model states".format(self._sphere_sensor_topic))

        return self.sphere_pos

    def _check_sphere_publishers_connection(self):
        """Checks that all the publishers are working.
        """
        # Check joint position controller publishers
        rate = rospy.Rate(10)  # 10hz
        while self._sphere_pos_publisher.get_num_connections() == 0 and not rospy.is_shutdown():
            rospy.loginfo("No susbribers to {} yet so we wait and try again".format(self._sphere_pose_topic))
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                # This is to avoid error when world is rested, time when backwards.
                pass
        while self._sphere_vel_publisher.get_num_connections() == 0 and not rospy.is_shutdown():
            rospy.loginfo("No susbribers to {} yet so we wait and try again".format(self._sphere_vel_topic))
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                # This is to avoid error when world is rested, time when backwards.
                pass
        rospy.logdebug("Sphere publisher connected!")
        # Check additional publishers
        rospy.logdebug("All publishers READY")

    def _gazebo_model_state_callback(self, data):
        self.gazebo_model_pose = dict(zip(data.name, data.pose))
        self.gazebo_model_twist = dict(zip(data.name, data.twist))