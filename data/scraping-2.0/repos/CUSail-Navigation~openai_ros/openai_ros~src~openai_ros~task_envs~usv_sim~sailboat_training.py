import rospy
import numpy
from gym import spaces
from openai_ros.robot_envs import usv_env
from gym.envs.registration import register
from geometry_msgs.msg import Point
from geometry_msgs.msg import Vector3
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64MultiArray
from tf.transformations import euler_from_quaternion
from openai_ros.task_envs.task_commons import LoadYamlFileParamsTest
from openai_ros.openai_ros_common import ROSLauncher
import os


class SailboatEnv(usv_env.USVSimEnv):

    def __init__(self):
        """
        Train a sailboat to move to the goal.
        """

        # This is the path where the simulation files, the Task and the Robot gits will be downloaded if not there
        ros_ws_abspath = rospy.get_param("/sailboat/training/ros_ws_abspath",
                                         None)
        assert ros_ws_abspath is not None, "You forgot to set ros_ws_abspath in your yaml file of your main RL script. Set ros_ws_abspath: \'YOUR/SIM_WS/PATH\'"
        assert os.path.exists(ros_ws_abspath), "The Simulation ROS Workspace path " + ros_ws_abspath + \
                                               " DOESNT exist, execute: mkdir -p " + ros_ws_abspath + \
                                               "/src;cd " + ros_ws_abspath + ";catkin_make"

        # TODO
        # ROSLauncher(rospackage_name="gazebo_ros",
        #             launch_file_name="empty_world.launch",
        #             ros_ws_abspath=ros_ws_abspath)

        # Load Params from the desired Yaml file
        LoadYamlFileParamsTest(
            rospackage_name="openai_ros",
            rel_path_from_package_to_file=
            "../../lib/python3/dist-packages/openai_ros/task_envs/usv_sim/config/",
            yaml_file_name="sailboat_training_config.yaml")

        # Here we will add any init functions prior to starting the MyRobotEnv
        super(SailboatEnv, self).__init__(ros_ws_abspath)

        rospy.loginfo("Start SailboatEnv INIT...")

        self.sail_angle = 0.0
        self.rudder_angle = 0.0
        self.prev_sail_angle = 0.0
        self.prev_rudder_angle = 0.0

        self.prev_x = None
        self.prev_y = None

        # We set the reward range, which is not compulsory but here we do it.
        self.reward_range = (-numpy.inf, numpy.inf)

        # number of steps per episode
        self.nsteps = rospy.get_param('/sailboat/sailboat_training/nsteps')

        # Actions and Observations
        # joint limits for clipping
        self.sail_upper_limit = rospy.get_param(
            '/sailboat/sailboat_training/sail_upper_limit')
        self.sail_lower_limit = rospy.get_param(
            '/sailboat/sailboat_training/sail_lower_limit')
        self.rudder_upper_limit = rospy.get_param(
            '/sailboat/sailboat_training/rudder_upper_limit')
        self.rudder_lower_limit = rospy.get_param(
            'sailboat_training/rudder_lower_limit')
        self.max_distance_from_goal = rospy.get_param(
            '/sailboat/sailboat_training/max_distance_from_goal')

        # end the episode if it goes outsides
        self.work_space_x_max = rospy.get_param(
            "/sailboat/training/work_space/x_max")
        self.work_space_x_min = rospy.get_param(
            "/sailboat/training/work_space/x_min")
        self.work_space_y_max = rospy.get_param(
            "/sailboat/training/work_space/y_max")
        self.work_space_y_min = rospy.get_param(
            "/sailboat/training/work_space/y_min")

        # threshold to goal
        self.goal_epsilon = rospy.get_param(
            "/sailboat/sailboat_training/goal_epsilon")

        # decimal precision of observations
        self.dec_obs = rospy.get_param(
            "/sailboat/sailboat_training/number_decimals_precision_obs")

        # Get the wind speed
        self.wind_x = rospy.get_param('/uwsim/wind/x')
        self.wind_y = rospy.get_param('/uwsim/wind/y')

        # We place the maximum and minimum values of observations

        # TODO what are the observations? For now let's say:
        # linear velocity x and y
        # angular velocity
        # joint positions of sail and rudder
        # relative wind vector x and y normalized so max magnitude is 1 (i.e. unit vector at rel wind angle)
        # distance from goal in x and y directions

        # TODO now its relative wind direction, yaw, dist x, dist y

        high = numpy.array([
            2 * numpy.pi, 2 * numpy.pi, self.max_distance_from_goal,
            self.max_distance_from_goal
        ])

        low = numpy.array([
            0.0, 0.0, 0.0, 0.0
        ])

        self.observation_space = spaces.Box(low, high)

        # Action space is just sail and rudder joints
        act_high = numpy.array(
            [self.sail_upper_limit, self.rudder_upper_limit])
        act_low = numpy.array([self.sail_lower_limit, self.rudder_lower_limit])
        self.action_space = spaces.Box(act_low, act_high)

        rospy.loginfo("ACTION SPACES TYPE===>" + str(self.action_space))
        rospy.loginfo("OBSERVATION SPACES TYPE===>" +
                      str(self.observation_space))

        # Rewards

        # reward for reaching the target
        self.done_reward = rospy.get_param(
            "/sailboat/sailboat_training/done_reward")

        # reward some multiple * vmg
        self.vmg_reward = rospy.get_param(
            "/sailboat/sailboat_training/vmg_reward_multiplier")

        # reward some multiple * dist_closer_to_goal
        self.dist_reward = rospy.get_param(
            '/sailboat/sailboat_training/dist_reward_multiplier')

        # penalize moving the joints too quickly/erratically
        # multiply penalty * displacement.norm()
        self.joint_penalty = rospy.get_param(
            "/sailboat/sailboat_training/joint_movement_penalty")

        # point deduction per step (to encourage finding the shortest path
        self.time_penalty = rospy.get_param(
            '/sailboat/sailboat_training/time_penalty')

        # stop the episode after 3 stalled (vmg=0) steps
        self.stall_kill = rospy.get_param(
            '/sailboat/sailboat_training/stall_kill')

        # time to wait between actions
        self.running_step = rospy.get_param(
            '/sailboat/sailboat_training/running_step')
        
        # Read the wind speed when it changes
        rospy.Subscriber('/sailboat/uwsim/wind_speed', Float64MultiArray, self.read_wind)

        rospy.loginfo("END SailboatEnv INIT...")

    def _set_init_pose(self):
        """
        Sets the two joints 0.0 and waits for the time_sleep
        to allow the action to be executed
        """

        self.set_joints(0.0, 0.0, time_sleep=1.0)

        return True

    def _init_env_variables(self):
        """
        Inits variables needed to be initialised each time we reset at the start
        of an episode.
        """

        # For Info Purposes
        self.total_reward = 0.0
        self.total_steps = 0
        self.stalled_steps = 0

        # We get the initial pose to mesure the distance from the desired point.
        odom = self.get_state()
        current_position = Vector3()
        current_position.x = odom.pose.pose.position.x
        current_position.y = odom.pose.pose.position.y

        self.x = current_position.x
        self.y = current_position.y
        self.prev_x = self.x
        self.prev_y = self.y

        x, y = self.get_distance_from_goal(current_position)

        self.prev_distance = numpy.sqrt(x**2 + y**2)

    def _set_action(self, action):
        """
        Sets the joints based on the action given.
        """

        sail_pos = action[0]
        rudder_pos = action[1]

        rospy.loginfo("Start Set Action ==> s: " + str(sail_pos) + ", r: " +
                      str(rudder_pos))

        self.prev_sail_angle = self.sail_angle
        self.prev_rudder_angle = self.rudder_angle

        # clip the joint positions to be within limits
        sail_pos = numpy.clip(sail_pos, self.sail_lower_limit,
                              self.sail_upper_limit)
        rudder_pos = numpy.clip(rudder_pos, self.rudder_lower_limit,
                                self.rudder_upper_limit)

        self.sail_angle = sail_pos
        self.rudder_angle = rudder_pos

        self.set_joints(sail_pos, rudder_pos, time_sleep=self.running_step)

        rospy.loginfo("END Set Action ==> s: " + str(sail_pos) + ', r: ' +
                      str(rudder_pos) + ". Slept for " +
                      str(self.running_step) + "s to execute movement")

    def _get_obs(self):
        """
        Here we define what sensor data defines our robots observations
        """
        rospy.loginfo("Start Get Observation ==>")

        # TODO what are the observations? For now let's say:
        # linear velocity x and y
        # angular velocity
        # joint positions of sail and rudder
        # rel unit wind vector x and y
        # distance from goal

        odom = self.get_state()
        base_position = odom.pose.pose.position
        base_orientation_quat = odom.pose.pose.orientation

        vel_x = (base_position.x - self.x) / self.running_step
        vel_y = (base_position.y - self.y) / self.running_step

        self.prev_x = self.x
        self.prev_y = self.y
        self.x = base_position.x
        self.y = base_position.y

        # TODO only using yaw for now, more params means more sim to real gap
        base_roll, base_pitch, base_yaw = self.get_orientation_euler(
            base_orientation_quat)
        base_speed_angular_yaw = odom.twist.twist.angular.z

        wind_rel_x, wind_rel_y = self.relative_wind_unit_vector(base_yaw)

        distance_x, distance_y = self.get_distance_from_goal(base_position)

        rel_wind_angle = self.relative_wind_angle(base_yaw)

        observation = []
        # observation.append(round(vel_x, self.dec_obs))
        # observation.append(round(vel_y, self.dec_obs))
        # observation.append(round(base_speed_angular_yaw, self.dec_obs))

        # observation.append(round(self.sail_angle, self.dec_obs))
        # observation.append(round(self.rudder_angle, self.dec_obs))

        # observation.append(round(wind_rel_x, self.dec_obs))
        # observation.append(round(wind_rel_y, self.dec_obs))

        observation.append(round(rel_wind_angle, self.dec_obs))
        observation.append(round(base_yaw, self.dec_obs))

        observation.append(round(distance_x, self.dec_obs))
        observation.append(round(distance_y, self.dec_obs))

        return observation

    def _is_done(self, observations):
        """
        We consider the episode done if:
        1) The sailboat is ouside the workspace
        2) It got to the desired point
        3) More than n steps have passed
        4) VMG is 0 for 3 steps
        """
        if self.total_steps >= self.nsteps:
            self.send_reset_signal()
            return True

        if self.stall_kill and (self.stalled_steps > 2):
            self.send_reset_signal()
            return True

        current_position = Vector3()
        current_position.x = self.x
        current_position.y = self.y

        is_inside_corridor = self.is_inside_workspace(current_position)
        has_reached_goal = self.is_in_desired_position(current_position,
                                                       self.goal_epsilon)
        if has_reached_goal:
            self.send_hit_waypoint()
            
        done = (not is_inside_corridor)
        if done:
            self.send_reset_signal()

        return done

    def _compute_reward(self, observations, done):
        """
        We base the rewards on:
        1) whether the sailboat has reached the goal
        2) the magnitude of the velocity made good (higher = better)
        3) how much the joint positions have moved
        """

        # We only consider the plane, the fluctuation in z is mainly due to waves
        current_position = Point()
        current_position.x = self.x
        current_position.y = self.y

        current_velocity = Point()
        current_velocity.x = (self.x - self.prev_x) / self.running_step
        current_velocity.y = (self.y - self.prev_y) / self.running_step

        rospy.loginfo("#### Compute Reward ####")
        rospy.loginfo("Current position: {}, {}".format(
            current_position.x, current_position.y))
        rospy.loginfo("Current velocity: {}, {}".format(
            current_velocity.x, current_velocity.y))
        rospy.loginfo("Goal: {}, {}".format(self.goal.pose.pose.position.x,
                                            self.goal.pose.pose.position.y))

        reached_goal = self.is_in_desired_position(current_position,
                                                   self.goal_epsilon)

        reward = 0.0

        # reward based on velocity made good
        goal = Point()
        goal.x = self.goal.pose.pose.position.x
        goal.y = self.goal.pose.pose.position.y

        vmg = self.velocity_made_good(current_velocity, current_position,
                                        goal)

        rospy.loginfo("VMG^3 IS {}".format(vmg**3))
        reward += self.vmg_reward * vmg**3

        if numpy.abs(vmg) < 1e-5:
            self.stalled_steps += 1
        else:
            self.stalled_steps = 0

        # reward based on moving closer to the goal
        dist = numpy.sqrt(observations[-2]**2 + observations[-1]**2)
        rospy.loginfo(
            'DECREASE IN DISTANCE IS {}'.format(self.prev_distance - dist))
        reward += self.dist_reward * (self.prev_distance - dist)
        self.prev_distance = dist

        # penalize based on the change in joint positions here
        sail_diff = self.sail_angle - self.prev_sail_angle
        rudder_diff = self.rudder_angle - self.prev_rudder_angle
        penalty = self.joint_penalty * (numpy.sqrt(sail_diff**2 +
                                                    rudder_diff**2))
        reward -= penalty
        rospy.loginfo("JOINT PENALTY IS {}".format(penalty))

        reward -= self.time_penalty

        if reached_goal:
            rospy.loginfo("Reached goal reward: {}".format(
                self.done_reward))
            reward = self.done_reward

        rospy.loginfo("reward=" + str(reward))
        self.total_reward += reward
        rospy.loginfo("Total reward=" + str(self.total_reward))
        self.total_steps += 1
        rospy.loginfo("Total steps=" + str(self.total_steps))

        return reward

    # Internal TaskEnv Methods

    def velocity_made_good(self, current_velocity, current_position,
                           goal_position):
        """
        Dot product of the current velocity (2D) and the displacement to goal
        (goal - current position)
        Use unit vectors to get invariance to wind speed
        """
        displacement = numpy.array([
            goal_position.x - current_position.x,
            goal_position.y - current_position.y
        ])

        dist = numpy.linalg.norm(displacement)
        displacement = displacement / dist if dist > 0.0 else numpy.zeros_like(
            displacement)

        vel = numpy.array([current_velocity.x, current_velocity.y])
        speed = numpy.linalg.norm(vel)
        vel = vel / speed if speed > 0.0 else numpy.zeros_like(vel)

        return numpy.dot(displacement, vel)

    def relative_wind_unit_vector(self, yaw):
        # all in radians
        abs_angle = numpy.arctan2(self.wind_y, self.wind_x)
        rel_angle = (abs_angle - yaw) % (2 * numpy.pi)

        # make unit vector
        x = numpy.cos(rel_angle)
        y = numpy.sin(rel_angle)
        return x, y
    
    def relative_wind_angle(self, yaw):
        # all in radians
        abs_angle = numpy.arctan2(self.wind_y, self.wind_x)
        rel_angle = (abs_angle - yaw) % (2 * numpy.pi)
        return rel_angle

    def get_distance_from_goal(self, current_position):
        goal_x = self.goal.pose.pose.position.x
        goal_y = self.goal.pose.pose.position.y

        displacement = numpy.array(
            [goal_x - current_position.x, goal_y - current_position.y])

        return displacement[0], displacement[1]

    def is_in_desired_position(self, current_position, epsilon=0.05):
        """
        Returns True if the current position is within epsilon of the goal
        """

        goal_x = self.goal.pose.pose.position.x
        goal_y = self.goal.pose.pose.position.y

        displacement = numpy.array(
            [goal_x - current_position.x, goal_y - current_position.y])

        dist = numpy.linalg.norm(displacement)

        reached = dist < epsilon

        # rospy.loginfo("###### IS WITHIN GOAL? ######")
        # rospy.loginfo("current_position" + str(current_position))
        # rospy.loginfo("distance: {}, epsilon {}, reached? {}".format(
        #     dist, epsilon, reached))
        # rospy.loginfo("############")

        return reached

    def get_orientation_euler(self, quaternion_vector):
        # We convert from quaternions to euler
        orientation_list = [
            quaternion_vector.x, quaternion_vector.y, quaternion_vector.z,
            quaternion_vector.w
        ]

        roll, pitch, yaw = euler_from_quaternion(orientation_list)
        return roll, pitch, yaw

    def is_inside_workspace(self, current_position):
        """
        Check if the boat is inside the defined workspace
        """
        is_inside = False

        # rospy.loginfo("##### INSIDE WORK SPACE? #######")
        # rospy.loginfo("XYZ current_position" + str(current_position))
        # rospy.loginfo("work_space_x_max" + str(self.work_space_x_max) +
        #               ",work_space_x_min=" + str(self.work_space_x_min))
        # rospy.loginfo("work_space_y_max" + str(self.work_space_y_max) +
        #               ",work_space_y_min=" + str(self.work_space_y_min))
        # rospy.loginfo("############")

        if current_position.x > self.work_space_x_min and current_position.x <= self.work_space_x_max:
            if current_position.y > self.work_space_y_min and current_position.y <= self.work_space_y_max:
                is_inside = True

        return is_inside

    def read_wind(self, msg):
        if len(msg.data):
            self.wind_x = msg.data[0]
            self.wind_y = msg.data[1]
            rospy.loginfo("### UPDATED WIND SPEED {} {} ###".format(self.wind_x, self.wind_y))
