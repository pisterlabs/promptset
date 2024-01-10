import os
from pickle import EMPTY_DICT
import numpy as np
import copy

from numpy.core.defchararray import join
import rospy
import time
from gym import spaces
from pyquaternion import Quaternion
from openai_ros.robot_envs.modrob_env_path_following import ModRobEnvPathFollowing
from gym.envs.registration import register
from geometry_msgs.msg import Point
from geometry_msgs.msg import Pose
from std_msgs.msg import Empty
from std_msgs.msg import Bool
from std_msgs.msg import Float64
from gazebo_msgs.msg import ModelStates, ModelState
from gazebo_msgs.srv import SetModelState
from openai_ros.task_envs.task_commons import LoadYamlFileParamsTest
from openai_ros.openai_ros_common import ROSLauncher
from modrob_simulation.msg import Collision, Collisions
from custom_robot_msgs.msg import PositionsHeadered

class ModRobSafeHumanEnv(ModRobEnvPathFollowing):
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

        ROSLauncher(rospackage_name="initialisation",
                    launch_file_name="start_world.launch",
                    ros_ws_abspath=ros_ws_abspath)

        # Load Params from the desired Yaml file
        LoadYamlFileParamsTest(rospackage_name="openai_ros",
                               rel_path_from_package_to_file="src/openai_ros/task_envs/human_modrob/config",
                               yaml_file_name="modrob_safe_human_random_" + self.robot_name_ + ".yaml")

        # Here we will add any init functions prior to starting the MyRobotEnv
        super(ModRobSafeHumanEnv, self).__init__(ros_ws_abspath)
        
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
        # Number of observations
        self.n_observations = len(self.observation_norm)
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.n_actions,))
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.n_observations,))
        rospy.logdebug("ACTION SPACES TYPE===>"+str(self.action_space))
        rospy.logdebug("OBSERVATION SPACES TYPE===>"+str(self.observation_space))

        self.cumulated_steps = 0.0

        ## Unpause sim and start all subscribers and publishers
        self.gazebo.unpauseSim()
        # Collision detection topic
        self._collision_topic = '/' + self.robot_name_ + '/collisions'
        
        # ROS subscriber to robot collision
        rospy.Subscriber(self._collision_topic, Collisions, self._collision_callback)
        
        self._human_joint_sensor_topic = '/human_joint_pos'
        # ROS subscriber and publisher for human animation 
        rospy.Subscriber(self._human_joint_sensor_topic, PositionsHeadered, self._human_joint_callback)
        self._check_human_joint_ready()
        self._init_human_animation_publisher()
        self._check_human_publishers_connection()

        ## Utility for moving start and goal position visual models
        # CAREFUL: The getter method has an s at the end while the setter method doesn't!
        # Ros subscriber to gazebo model state
        self._gazebo_model_state_topic = '/gazebo/model_states'
        rospy.Subscriber(self._gazebo_model_state_topic, ModelStates, self._gazebo_model_state_callback)
        self._check_gazebo_model_state_ready()
        # Get the base pose
        self.base_pose = self._get_base_pose()
        self._send_initialization()
        is_init = None
        while is_init is None and not rospy.is_shutdown():
            try:
                is_init = rospy.wait_for_message("/initialisation", Empty, timeout=5.0)
                rospy.logdebug("Current {} READY=>".format("/initialisation"))

            except:
                rospy.logerr("Current {} not ready yet, retrying for getting joint states".format("/initialisation"))
        self.gazebo.pauseSim()

    def _get_env_variables(self):
        """Load in environment variables from yaml.
        Relevant variables:
            joint_min: Minimal angle for all joints
            joint_max: Maximal angle for all joints
            joint_max_delta: Max theoretically allowed movement per execution step (in rad)
            joint_max_v: Maximal velocity of robot joints (for observation only)
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
            goal_error: 0.1 # Precision for goal reached
            init_joint_position: [joint1, ..., jointN] initial position
            init_hand_to_finger1_position: Initial position for gripper part 1 - TODO: Change this to dynamically.
            init_hand_to_finger2_position: Initial position for gripper part 2 - TODO: Change this to dynamically.
            use_goal_randomization: If false: use fix goal_pose, if true: use random position in goal_area
            goal_pose: geometry_msgs.msg.Pose for end effector goal
            goal_area: min and max values for goal position
            distance_penalty_position: Reward penalty for position distance
            distance_penalty_orientation: Reward penalty for orientation distance
            time_penalty: Time penalty for every step
            goal_reward: Points given when reaching the goal
            collision_penalty: Penalty when colliding with an object
            critical_collision_penalty: Penalty when critically colliding with human
            motion_capture/* : Motion capture information. See config in human_reach package.
            human_motion_pos_random: Randomize position of animation uniformly by +/- this value
            human_motion_time_random: Randomize the starting time of the animation uniformly by this value [0; val]
            safety_distance_ground: Distance to ground for collision checking
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
        self.joint_max_v = rospy.get_param('/modrob/joint_max_v')
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
            # for current joint values
            self.observation_norm.append([self.joint_min, self.joint_max-self.joint_min])
            # for goal joint values
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
        # defines where the current joint values values start in observation
        self.observation_id_current_joint_values = 0
        self.observation_id_goal_joint_values = len(self.get_arm_joint_names())
        self.observation_id_current_joint_velocities = len(self.observation_norm)
        for joint_name in self.get_arm_joint_names():
          self.observation_norm.append([-self.joint_max_v, 2*self.joint_max_v])
        # Add normalization for current ee position
        self.observation_id_current_ee_pose = len(self.observation_norm) # defines where the current ee pose values start in observation
        # We don't normalize cartesian positions anymore.
        # You can experiment with this.
        """
        ee_limits = rospy.get_param('/modrob/ee_limits')
        self.observation_norm.append([ee_limits.get("min").get("x"), ee_limits.get("max").get("x")-ee_limits.get("min").get("x")])
        self.observation_norm.append([ee_limits.get("min").get("y"), ee_limits.get("max").get("y")-ee_limits.get("min").get("y")])
        self.observation_norm.append([ee_limits.get("min").get("z"), ee_limits.get("max").get("z")-ee_limits.get("min").get("z")])
        """
        self.observation_norm.append([0, 1]) #x: We don't normalize this
        self.observation_norm.append([0, 1]) #y: We don't normalize this
        self.observation_norm.append([0, 1]) #z: We don't normalize this
        # We start with just the wrist position as human information.
        # This can be extended by other human measurements.
        # It should be investigated if a recurrent network structure benefits the RL in this area.
        human_joint_names = rospy.get_param('/motion_capture/joint_names')
        human_extremities = rospy.get_param('/motion_capture/extremities')
        # The id of relevant joints in the motion capture measurements
        self.human_joint_meas_ids = []
        self.observation_id_human_joints = len(self.observation_norm) 
        for extremity in human_extremities:
            key = next(iter(extremity))
            # An extremity in the config has 3 elements - the last one is the extremity itself
            wrist = extremity[key][2]
            wrist_joint_name = wrist[next(iter(wrist))]
            assert wrist_joint_name in human_joint_names
            self.human_joint_meas_ids.append(human_joint_names.index(wrist_joint_name))
            self.observation_norm.append([0, 1]) #x: We don't normalize this
            self.observation_norm.append([0, 1]) #y: We don't normalize this
            self.observation_norm.append([0, 1]) #z: We don't normalize this
        # Add human head position
        self.human_joint_meas_ids.append(human_joint_names.index("head"))
        self.observation_norm.append([0, 1]) #x: We don't normalize this
        self.observation_norm.append([0, 1]) #y: We don't normalize this
        self.observation_norm.append([0, 1]) #z: We don't normalize this
        self.n_human_obs = 3
        # Add one entry that indicates collisions
        self.observation_id_collision = len(self.observation_norm)
        self.observation_norm.append([0, 1])
        self.observation_id_critical_collision = len(self.observation_norm)
        self.observation_norm.append([0, 1])
        

        self.action_norm = np.array(self.action_norm)
        self.observation_norm = np.array(self.observation_norm)
        
        # Movement settings
        self.movement_error = rospy.get_param('/modrob/movement_error')
        self.movement_timeout = rospy.get_param('/modrob/movement_timeout')
        self.init_error = rospy.get_param('/modrob/init_error')
        self.goal_error = rospy.get_param('/modrob/goal_error')
        # Set initial joint positions
        # Right now, only arm position movement implemented!
        # TODO: Add init gripper position
        self.init_arm_joint_position = []
        if rospy.has_param("/modrob/init_joint_position"):
            self.init_arm_joint_position = rospy.get_param("/modrob/init_joint_position")
        assert(len(self.init_arm_joint_position) == len(self._arm_joint_names))
        # Goal pose
        self.use_goal_randomization = False
        if (rospy.has_param('/modrob/use_goal_randomization')):
          self.use_goal_randomization = rospy.get_param('/modrob/use_goal_randomization')
        if self.use_goal_randomization:
          self.min_goal_pos_x = rospy.get_param('/modrob/goal_area/position/x_min')
          self.min_goal_pos_y = rospy.get_param('/modrob/goal_area/position/y_min')
          self.min_goal_pos_z = rospy.get_param('/modrob/goal_area/position/z_min')
          self.max_goal_pos_x = rospy.get_param('/modrob/goal_area/position/x_max')
          self.max_goal_pos_y = rospy.get_param('/modrob/goal_area/position/y_max')
          self.max_goal_pos_z = rospy.get_param('/modrob/goal_area/position/z_max')
          if rospy.has_param('modrob/goal_joint_position'):
            self.goal_arm_joint_position = rospy.get_param('modrob/goal_joint_position')
            self.goal_joint_diff = rospy.get_param('/modrob/goal_area/joint_diff')
          else:
            rospy.logwarn("Parameter modrob/goal_joint_position not found.")
            self.goal_arm_joint_position = [0.0 for _ in range(len(self._arm_joint_names))]
            self.goal_joint_diff = 0.0
        else:
          self.min_goal_pos_x = -1000
          self.min_goal_pos_y = -1000
          self.min_goal_pos_z = -1000
          self.max_goal_pos_x = 1000
          self.max_goal_pos_y = 1000
          self.max_goal_pos_z = 1000
          if rospy.has_param('modrob/goal_joint_position'):
            rospy.logwarn("Parameter modrob/goal_joint_position not found.")
            self.goal_arm_joint_position = rospy.get_param('modrob/goal_joint_position')
            self.goal_joint_diff = 0.0
          else:
            self.goal_arm_joint_position = [0.0 for _ in range(len(self._arm_joint_names))]
            self.goal_joint_diff = 0.0
          

        # Rewards
        self.distance_penalty_position = rospy.get_param('/modrob/distance_penalty_position')
        self.distance_penalty_orientation = rospy.get_param('/modrob/distance_penalty_orientation')
        self.time_penalty = rospy.get_param('/modrob/time_penalty')
        self.goal_reward = rospy.get_param('/modrob/goal_reward')
        self.collision_penalty = rospy.get_param('/modrob/collision_penalty')
        self.critical_collision_penalty = rospy.get_param('/modrob/critical_collision_penalty')
        
        # Human motion animation
        self.human_motion_pos_random = rospy.get_param('/modrob/human_motion_pos_random') 
        self.human_motion_time_random = rospy.get_param('/modrob/human_motion_time_random') 
        self.safety_distance_ground = rospy.get_param('/modrob/safety_distance_ground') 


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
        # Move the human out of the way
        self._human_play_publisher.publish(False)
        self._human_pose_shift_publisher.publish(self.create_pose([5, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]))
        # Move until init position is reached (timeout=0)
        success = False
        while (not success):
          success = self.move_arm_joints(self.init_arm_joint_position, error=self.init_error, timeout=0.0)
        rospy.sleep(0.1)
        self.init_pose = self.get_ee_pose()
        return True

    def _get_state_from_obs(self, obs):
      """Extract the state from an observation."""
      return obs[self.observation_id_current_joint_values:self.observation_id_goal_joint_values]

    def _get_goal_from_obs(self, obs):
      """Extract the goal from an observation."""
      return obs[self.observation_id_goal_joint_values:self.observation_id_current_joint_velocities]

    def _get_collision_from_obs(self, obs):
      """Extract the collision value from an observation."""
      return obs[self.observation_id_collision]

    def _get_critical_collision_from_obs(self, obs):
      """Extract the information if a collision was critical from an observation."""
      return obs[self.observation_id_critical_collision]

    def _replace_goal_in_obs(self, obs, new_goal):
      obs[self.observation_id_goal_joint_values:self.observation_id_current_joint_velocities] = new_goal
      return obs

    def _init_env_variables(self):
        """Inits episode specific variables each time we reset at the start of an episode.
        """
        # For Info Purposes
        self.cumulated_reward = 0.0
        # Set to false Done, because its calculated asyncronously
        self._episode_done = False
        # A counter for timesteps at goal position (Prevents overshooting of goal)
        self.goal_count = 0
        self.last_arm_joint_position = self.get_arm_joint_positions()
        self.is_collided = False
        self.critically_collided = False
        # Generate new random goal pose
        if self.create_random_goal_pose:
          self.goal_pose, goal_joint_values = self.create_random_goal_pose()
          self.goal_observation = goal_joint_values
        # Place start and goal markers
        self.move_gazebo_model("start_pos", self.correct_model_pose(self.init_pose))
        self.move_gazebo_model("goal_pos", self.correct_model_pose(self.goal_pose))
        # Set the human animation start position and time
        new_x = np.random.uniform(low = -self.human_motion_pos_random, high = self.human_motion_pos_random)
        new_y = np.random.uniform(low = -self.human_motion_pos_random, high = self.human_motion_pos_random)
        self._human_pose_shift_publisher.publish(self.create_pose([new_x, new_y, 0.0], [0.0, 0.0, 1.0, 0.0]))
        new_t = np.random.uniform(low = 0, high= self.human_motion_time_random)
        self._human_script_time_publisher.publish(new_t)
        self._human_play_publisher.publish(True)
        rospy.loginfo("Init env variables finished")
        

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
        
        # If commanded action would result in the end effector colliding with the ground,
        # don't execute that action.
        while(self.check_collision_ground(joint_positions, self.safety_distance_ground)):
          # Collision with ground highly likely
          rospy.loginfo("Prevented collision with ground. Will create new action.")
          rand_act = np.random.rand(self.n_actions)
          action = rand_act * (self.action_space.high-self.action_space.low) + self.action_space.low
          denormalized_action = self.denormalize_actions(action)
          if self.use_delta_actions:
            joint_positions = self.create_joint_positions_delta(denormalized_action)
          else:
            joint_positions = self.create_joint_positions_absolute(denormalized_action)

        # If the commanded action is withing the goal reached error bound,
        # move to the exact goal and don't stop before.
        # The idea behind this is to mitigate the effects of randomness around the goal
        # and ensure exactness.
        # TODO: Test this feature for praticability
        """
        if (self.joints_close(joint_positions, self.goal_observation, self.goal_error)):
          joint_positions = self.goal_observation
          movement_err = self.init_error
          timeout = 0
        """
        # Set action as command
        movement_err = self.movement_error
        timeout = self.movement_timeout
        # Only arm movement implemented right now. TODO: Add gripper action.
        #rospy.loginfo('New goal = [' + (' '.join('{}'.format(jp) for jp in joint_positions)) + ']')
        self.move_arm_joints(joint_positions, error=movement_err, timeout=timeout)
        rospy.logdebug("END Set Action ==>"+str(action))
        return action

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
        if self._get_collision_from_obs(observations):
            return True
        return self._is_in_goal(observations)
        

    def _is_in_goal(self, obs):
      """Return if the observed state is in the goal position."""
      observations = self.denormalize_observations(obs)
      # Check if joint position is close
      joint_values = self._get_state_from_obs(observations)
      goal_joint_values = self._get_goal_from_obs(observations)
      if (self.joints_close(joint_values, goal_joint_values, self.goal_error)):
          return True
      else:
          return False

    def _compute_reward(self, observations, done):
        """Compute sparse reward for this step."""
        if self._get_critical_collision_from_obs(observations):
            return self.critical_collision_penalty
        # If collision -> Return -1
        if self._get_collision_from_obs(observations):
            return self.collision_penalty
        # If goal -> Return 0
        if self._is_in_goal(observations):
            return self.goal_reward
        # If not done -> Return -1
        return self.time_penalty


    def _collision_callback(self, data):
        """This function is called if a collision is detected by one of the modrobs collision sensors.
        The collision sensor plugins publish the collision information on /' + self.robot_name_ + '/collisions.
        Sets self.is_collided to true.
        Outputs an info message.
        """
        if not self.is_collided:
            self.is_collided = True
            if (data.collisions[0].obstacle_contact.find("actor") >= 0 and 
                np.sum(np.abs(self.joint_state.velocity[2:]))/(len(self.joint_state.velocity)-2) > 0.1):
              rospy.logerr("Critical collision detected between {} and {} with velocity [{}].".format(
                  data.collisions[0].parent_contact, 
                  data.collisions[0].obstacle_contact,
                  self.joint_state.velocity[2:]
              ))
              self.critically_collided = True
            else:
              rospy.logwarn("Non-critical collision detected between {} and {} with velocity [{}].".format(
                  data.collisions[0].parent_contact, 
                  data.collisions[0].obstacle_contact,
                  self.joint_state.velocity[2:]
              )) 
              self.critically_collided = False
    
    
    def _human_joint_callback(self, data):
        """Incoming human joint (motion capture) measurement."""
        for (i, j_id) in enumerate(self.human_joint_meas_ids):
            self.human_joint_pos[i] = [data.data[j_id].x, data.data[j_id].y, data.data[j_id].z]

    def _init_human_animation_publisher(self):
      """Initialize the ROS topics to control the human animation.
      
      Possible actions are:
        /actor/pose_cmd: offset the position
        /actor/script_time_cmd: set the animation time
        /actor/start_stop_cmd: start/stop
      """
      self._human_pose_shift_publisher = rospy.Publisher('/actor/pose_cmd', Pose, queue_size=100)
      self._human_script_time_publisher = rospy.Publisher('/actor/script_time_cmd', Float64, queue_size=100)
      self._human_play_publisher = rospy.Publisher('/actor/start_stop_cmd', Bool, queue_size=100)
    
    def _check_human_publishers_connection(self):
      """Check that all the human publishers are working."""
      # Check joint position controller publishers
      rate = rospy.Rate(10)  # 10hz
      while self._human_pose_shift_publisher.get_num_connections() == 0 and not rospy.is_shutdown():
        rospy.logdebug("No susbribers to /actor/pose_cmd yet so we wait and try again")
        try:
          rate.sleep()
        except rospy.ROSInterruptException:
          # This is to avoid error when world is rested, time when backwards.
          pass
      while self._human_script_time_publisher.get_num_connections() == 0 and not rospy.is_shutdown():
        rospy.logdebug("No susbribers to /actor/script_time_cmd yet so we wait and try again")
        try:
          rate.sleep()
        except rospy.ROSInterruptException:
          # This is to avoid error when world is rested, time when backwards.
          pass
      while self._human_play_publisher.get_num_connections() == 0 and not rospy.is_shutdown():
        rospy.logdebug("No susbribers to /actor/start_stop_cmd yet so we wait and try again")
        try:
          rate.sleep()
        except rospy.ROSInterruptException:
          # This is to avoid error when world is rested, time when backwards.
          pass
      rospy.logdebug("All human control publishers connected!")

    def create_random_goal_pose(self):
      """Randomly sample joint value poses until one lays in the goal area.
      
      Return:
        valid joint values so that ee position in goal area
      """
      t = time.time()
      while True:
        ee_pose, goal_joint_values = self.get_random_joint_pose(self.goal_arm_joint_position, self.goal_joint_diff)
        if (ee_pose.position.x >= self.min_goal_pos_x and
            ee_pose.position.x <= self.max_goal_pos_x and
            ee_pose.position.y >= self.min_goal_pos_y and
            ee_pose.position.y <= self.max_goal_pos_y and
            ee_pose.position.z >= self.min_goal_pos_z and
            ee_pose.position.z <= self.max_goal_pos_z and
            not self.check_collision_ground(goal_joint_values, self.safety_distance_ground)):
          elapsed = time.time() - t
          rospy.loginfo("Needed {} s to calculate new goal.".format(elapsed))
          return ee_pose, goal_joint_values

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
            - goal_joint_positions
            - current ee position (cartesian)
            - human obstacle position relative to ee position (cartesian)
            - is collided
        Returns:
            observations (list): non normalized observations, shape = [n_observations]
        """
        # last_arm_joint_position = current joint position after env step
        self.last_arm_joint_position = self.get_arm_joint_positions()
        arm_joint_velocities = self.get_arm_joint_velocities()
        ee_position, _ = self.get_current_ee_position_and_quaternion()
        observations = self.last_arm_joint_position + self.goal_observation + arm_joint_velocities + ee_position 
        for i in range(self.n_human_obs):
          for j in range(3):
            observations.append(self.human_joint_pos[i][j] - ee_position[j])
        if self.is_collided:
          observations.append(1)
          if self.critically_collided:
            observations.append(1)
          else:
            observations.append(0)
        else:
          observations.append(0)
          observations.append(0)
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
        rospy.logdebug("Position distance = {}, orientation distance = {}".format(position_distance, orientation_distance))
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
                rospy.logerr("Current {} not ready yet, retrying for getting gazebo_model states.".format(self._gazebo_model_state_topic))

        return self.gazebo_model_pose

    
    def _check_human_joint_ready(self):
        self.human_joint_pos = []
        for _ in range(len(self.human_joint_meas_ids)):
            self.human_joint_pos.append([])
        rospy.logdebug("Waiting for {} to be READY...".format(self._human_joint_sensor_topic))
        while len(self.human_joint_pos[0]) == 0 and not rospy.is_shutdown():
            try:
                data = rospy.wait_for_message(self._human_joint_sensor_topic, PositionsHeadered, timeout=5.0)
                self._human_joint_callback(data)
                rospy.logdebug("Current {} READY=>".format(self._human_joint_sensor_topic))

            except:
                rospy.logerr("Current {} not ready yet, retrying for getting human motion capture information.".format(self._human_joint_sensor_topic))

        return self.human_joint_pos
    

    
    def _check_sphere_publishers_connection(self):
        """
        #Checks that all the publishers are working.
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
        """
        return True

    def _gazebo_model_state_callback(self, data):
        self.gazebo_model_pose = dict(zip(data.name, data.pose))
        self.gazebo_model_twist = dict(zip(data.name, data.twist))
   