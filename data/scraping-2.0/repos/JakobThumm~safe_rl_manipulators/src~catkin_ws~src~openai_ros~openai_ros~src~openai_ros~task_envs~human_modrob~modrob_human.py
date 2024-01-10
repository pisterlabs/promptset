import rospy
import numpy as np
from gym import spaces
from openai_ros.robot_envs import modrob_env
from gym.envs.registration import register
from geometry_msgs.msg import Point
from openai_ros.task_envs.task_commons import LoadYamlFileParamsTest
from openai_ros.openai_ros_common import ROSLauncher
import os

class ModRobHumanEnv(modrob_env.ModRobEnv):
    def __init__(self):
        """
        This Task Env is designed for having the ModRob in a human working environment.
        It will learn how to finish tasks without colliding with the human.
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
                    launch_file_name="start_world_human.launch",
                    ros_ws_abspath=ros_ws_abspath)

        # Load Params from the desired Yaml file
        LoadYamlFileParamsTest(rospackage_name="openai_ros",
                               rel_path_from_package_to_file="src/openai_ros/task_envs/human_modrob/config",
                               yaml_file_name="modrob_human.yaml")

        # Here we will add any init functions prior to starting the MyRobotEnv
        super(ModRobHumanEnv, self).__init__(ros_ws_abspath)
        
        ## Load in environment variables
        self._get_env_variables()
        # We set the reward range, which is not compulsory but here we do it.
        self.reward_range = (-np.inf, np.inf)

        ## Set action and observation space
        # Continuous action space.
        # All actions should range from 0 to 1. This improves training.
        self.n_actions = len(self.id_action)
        self.n_observations = len(self.id_observation)
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.n_actions,))
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.n_observations,))
        rospy.logdebug("ACTION SPACES TYPE===>"+str(self.action_space))
        rospy.logdebug("OBSERVATION SPACES TYPE===>"+str(self.observation_space))

        self.cumulated_steps = 0.0

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
            use_delta_actions: True: Use differential position commands, False: Use absolute position commands
            movement_error: Precision maximum for regular movement (can be arbitrarely small)
            movement_timeout: Time waited until next movement execution
            init_error: Precision for start position
            goal_error: Precision for goal reached
            init_joint0_position: Initial position for joint0 - TODO: Change this to dynamically.
            init_joint1_position: Initial position for joint1 - TODO: Change this to dynamically.
            init_joint2_position: Initial position for joint2 - TODO: Change this to dynamically.
            init_hand_to_finger1_position: Initial position for gripper part 1 - TODO: Change this to dynamically.
            init_hand_to_finger2_position: Initial position for gripper part 2 - TODO: Change this to dynamically.
            desired_pose: Dummy pose for joint0 - TODO: Change this to dynamically.
            distance_reward: Getting closer to the reward gives positive reward
            time_penalty: Time penalty for every step
            goal_reward: Points given when reaching the goal
        """
        ## Determine the normalization constants for all observations and actions.
        # action_norm consists of two columns (c, r) and one row per controllable joint.
        # To normalize an action a, calculate: a_n = (a-c)/r, r!=0
        # To denormalize an normalized action a_n, calculate a = (a_n * r) + c
        # Since we are using the tanh as activation for actions, we normalize a to [-1; 1]
        self.action_norm = []
        self.observation_norm = []
        # Additionally, define which id refers to which action and observation.
        self.id_action = []
        self.id_observation = []
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
        for joint_name in self.get_joint_names():
            if "joint" in joint_name:
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
                self.id_action.append(joint_name)
                self.id_observation.append(joint_name)
                # TEST for dummy reward and goal
                if joint_name == "joint0":
                    self.goal_joint_id = len(self.id_observation)-1
            elif "hand_to_finger1" in joint_name:
                # Only one action for both grippers (they move together)
                if self.use_delta_actions:
                    self.action_norm.append([-1*self.gripper_max_delta, 2*self.gripper_max_delta])
                else:
                    self.action_norm.append([self.gripper1_min, self.gripper1_max-self.gripper1_min])
                self.observation_norm.append([self.gripper1_min, self.gripper1_max-self.gripper1_min])
                self.id_action.append(joint_name)
                self.id_observation.append(joint_name)
        self.action_norm = np.array(self.action_norm)
        self.observation_norm = np.array(self.observation_norm)
        # Movement settings
        self.movement_error = rospy.get_param('/modrob/movement_error')
        self.movement_timeout = rospy.get_param('/modrob/movement_timeout')
        self.init_error = rospy.get_param('/modrob/init_error')
        self.goal_error = rospy.get_param('/modrob/goal_error')
        self.init_arm_joint_position = []
        if rospy.has_param("/modrob/init_joint_position"):
            self.init_arm_joint_position = rospy.get_param("/modrob/init_joint_position")
        assert(len(self.init_arm_joint_position) == len(self._arm_joint_names))
        
        # Goal and reward
        self.desired_pose = rospy.get_param('/modrob/desired_pose')
        self.distance_reward = rospy.get_param('/modrob/distance_reward')
        self.time_penalty = rospy.get_param('/modrob/time_penalty')
        self.goal_reward = rospy.get_param('/modrob/goal_reward')


    def _set_init_pose(self):
        """Sets the Robot in its init pose.
        """
        # Move until init position is reached (timeout=0)
        self.move_all_joints(self.init_joint_position, error=self.init_error, timeout=0.0)     
        return True


    def _init_env_variables(self):
        """Inits episode specific variables each time we reset at the start of an episode.
        """
        # For Info Purposes
        self.cumulated_reward = 0.0
        # Set to false Done, because its calculated asyncronously
        self._episode_done = False
        self.last_joint_position = self.get_joint_positions()


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
        self.move_all_joints(joint_positions, error=self.movement_error, timeout=self.movement_timeout)
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
        """Compute if episode is finished.
        Right now only dummy goal for first joint angle.
        """
        observations = self.denormalize_observations(observations)
        phi1 = observations[self.goal_joint_id]
        if np.isclose(phi1, self.desired_pose, atol=self.goal_error):
            self._episode_done = True
        return self._episode_done

    def _compute_reward(self, observations, done):
        """Compute reward for this step.
        Right now only dummy reward for first joint angle.
        """
        reward = 0
        observations = self.denormalize_observations(observations)
        phi1 = observations[self.goal_joint_id]
        if not done:
            # Reward of minus 1 per time step
            reward -= self.time_penalty
            # Reward for getting closer to desired pos
            reward -= self.distance_reward * (self.desired_pose-phi1)
        else:
            reward += self.goal_reward

        rospy.logdebug("This step reward = " + str(reward))
        self.cumulated_reward += reward
        rospy.logdebug("Cumulated_reward = " + str(self.cumulated_reward))
        self.cumulated_steps += 1
        rospy.logdebug("Cumulated_steps = " + str(self.cumulated_steps))
        return reward

    def create_joint_positions_absolute(self, actions):
        """Creates joint_positions from an absolute action array.
        Args:
            actions: Action array (This should be denormalized!), shape = [n_actions]
        Returns:
            joint_positions (Dict): key: joint_name, value: desired joint position.
        """
        joint_positions = dict()
        for i, action_name in enumerate(self.id_action):
            if "joint" in action_name:
                joint_positions[action_name] = np.clip(actions[i], self.joint_min, self.joint_max)
            elif "hand_to_finger" in action_name:
                joint_positions[action_name] = np.clip(actions[i], self.gripper1_min, self.gripper1_max)
                # Set action for finger 2 reversed to finger 1.
                # F1 = min --> F2 = max; F1 = max --> F2 = max
                # F2 = ax + b
                a = (self.gripper2_min - self.gripper2_max)/(self.gripper1_max - self.gripper1_min)
                b = (self.gripper1_max*self.gripper2_max-self.gripper1_min*self.gripper2_min)/(self.gripper1_max - self.gripper1_min)
                joint_positions["hand_to_finger2"] = a * actions[i] + b
        return joint_positions

    def create_joint_positions_delta(self, actions):
        """Creates absolute joint_positions from an delta action array.
        Args:
            actions: Action array (This should be denormalized!), shape = [n_actions]
        Returns:
            joint_positions (Dict): key: joint_name, value: desired absolute joint position.
        """
        # Use the last observed joint position (not the current!)
        last_joint_positions = self.last_joint_position
        joint_positions = dict()
        for i, joint_name in enumerate(self.id_action):
            if "joint" in joint_name:
                # Calculate new desired joint position and keep it in joint ranges
                joint_positions[joint_name] = np.clip(last_joint_positions[joint_name] + actions[i], self.joint_min, self.joint_max)
            elif "hand_to_finger" in joint_name:
                joint_positions[joint_name] = np.clip(last_joint_positions[joint_name] + actions[i], self.gripper1_min, self.gripper1_max)
                # Set action for finger 2 reversed to finger 1.
                # F1 = min --> F2 = max; F1 = max --> F2 = max
                # F2 = ax + b
                a = (self.gripper2_min - self.gripper2_max)/(self.gripper1_max - self.gripper1_min)
                b = (self.gripper1_max*self.gripper2_max-self.gripper1_min*self.gripper2_min)/(self.gripper1_max - self.gripper1_min)
                joint_positions["hand_to_finger2"] = a * joint_positions[joint_name] + b
        return joint_positions

    def retrieve_observations(self):
        """Retrieve all observations (not normalized).
        Converts joint_positions (Dict): key: joint_name, value: desired joint position, to observation array.
        Returns:
            observations (np.array): non normalized observations, shape = [n_observations]
        """
        self.last_joint_position = self.get_joint_positions()
        observations = np.zeros([len(self.id_observation)])
        for i, observation_name in enumerate(self.id_observation):
            if "joint" in observation_name:
                observations[i] = self.last_joint_position[observation_name]
            elif "hand_to_finger1" in observation_name:
                # Only use one gripper observation
                observations[i] = self.last_joint_position[observation_name]
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


    def is_in_desired_position(self,current_position, epsilon=0.05):
        """
        It return True if the current position is similar to the desired poistion
        """

        is_in_desired_pos = False


        x_pos_plus = self.desired_point.x + epsilon
        x_pos_minus = self.desired_point.x - epsilon
        y_pos_plus = self.desired_point.y + epsilon
        y_pos_minus = self.desired_point.y - epsilon

        x_current = current_position.x
        y_current = current_position.y

        x_pos_are_close = (x_current <= x_pos_plus) and (x_current > x_pos_minus)
        y_pos_are_close = (y_current <= y_pos_plus) and (y_current > y_pos_minus)

        is_in_desired_pos = x_pos_are_close and y_pos_are_close

        return is_in_desired_pos


    def get_distance_from_desired_point(self, current_position):
        """
        Calculates the distance from the current position to the desired point
        :param start_point:
        :return:
        """
        distance = self.get_distance_from_point(current_position,
                                                self.desired_point)

        return distance

    def get_distance_from_point(self, pstart, p_end):
        """
        Given a Vector3 Object, get distance from current position
        :param p_end:
        :return:
        """
        a = np.array((pstart.x, pstart.y, pstart.z))
        b = np.array((p_end.x, p_end.y, p_end.z))

        distance = np.linalg.norm(a - b)

        return distance
