#!/usr/bin/env python
from gym import utils
import copy
import math
import rospy
import rospkg
from gym import spaces
from openai_ros.robot_envs import ur5_lab_env
import numpy as np
from sensor_msgs.msg import JointState
from openai_ros.openai_ros_common import ROSLauncher
from openai_ros.task_envs.task_commons import LoadYamlFileParamsTest
import os
from gazebo_msgs.srv import SpawnModel, SpawnModelRequest, GetModelState, DeleteModel, GetWorldProperties
from geometry_msgs.msg import Point, Quaternion, Pose


class UR5LabSimTask(ur5_lab_env.UR5LabSimEnv, utils.EzPickle):
    def __init__(self):
        # Load Params from the desired Yaml file relative to this TaskEnvironment
        LoadYamlFileParamsTest(rospackage_name="openai_ros",
                               rel_path_from_package_to_file="src/openai_ros/task_envs/ur5_lab/config",
                               yaml_file_name="simple_task.yaml")

        # This is the path where the simulation files are

        ros_ws_abspath = rospy.get_param("ros_ws_path", None)
        start_gazebo = rospy.get_param("start_gazebo", True)
        visualize = rospy.get_param("visualize", False)

        ROSLauncher(rospackage_name="ur_gazebo",
                    launch_file_name="ur5_lab_world.launch",
                    launch_arguments=dict(gui=visualize, start_gazebo=start_gazebo),
                    ros_ws_abspath=ros_ws_abspath)

        super(UR5LabSimTask, self).__init__(ros_ws_abspath)

        rospy.logdebug("Entered UR5LabSimTask Env")
        self.get_params()

        self.action_space = spaces.Box(low=-0.1, high=0.1, shape=(6,))

        self.n_observations = 6
        self.observation_space = spaces.Box(low=-2, high=2, shape=(6,))
        self._draw_goal_pos()
        self.init_distance = None

    def get_params(self):
        # get configuration parameters

        init_pos_dict = rospy.get_param('/ur5_lab/init_pos')
        self.init_pos = [init_pos_dict["shoulder_pan_joint"],
                         init_pos_dict["shoulder_lift_joint"],
                         init_pos_dict["elbow_joint"],
                         init_pos_dict["wrist_1_joint"],
                         init_pos_dict["wrist_2_joint"],
                         init_pos_dict["wrist_3_joint"],
                         ]

        goal_pos_dict = rospy.get_param('/ur5_lab/goal_pos')
        self.goal_pos = [goal_pos_dict["x_pos"],
                         goal_pos_dict["y_pos"],
                         goal_pos_dict["z_pos"]]

        self.reached_goal_reward = rospy.get_param(
            '/ur5_lab/reached_goal_reward')

    def _set_init_pose(self):
        """
        Sets the Robot in its init pose
        """
        # Check because it seems its not being used
        rospy.logdebug("Init Pos:")
        rospy.logdebug(self.init_pos)

        # INIT POSE
        rospy.logdebug("Moving To Init Pose ")
        self.move_joints(self.init_pos)
        self.last_action = "INIT"

        pose = self.get_ee_pose()
        pose = np.array([pose.position.x,
                         pose.position.y,
                         pose.position.z, ])

        self.init_distance = np.linalg.norm(np.array(pose[:]) - np.array(self.goal_pos[:]))

        return True

    def _create_ball_request(self, px, py, pz):
        target_path = rospkg.RosPack()
        target_path = target_path.get_path('ur_description')
        target_path += '/urdf/target/target_point_Blue.urdf'
        target_urdf = None

        with open(target_path, "r") as g:
            target_urdf = g.read()

        target_pose = Pose(Point(x=px, y=py, z=pz), Quaternion(x=0, y=0, z=0, w=1))

        req = SpawnModelRequest()
        req.model_name = "target_ball"
        req.model_xml = target_urdf
        req.initial_pose = target_pose

        return req

    def _check_existence(self):
        get_world_specs = rospy.ServiceProxy('/gazebo/get_world_properties', GetWorldProperties)()
        model_names = get_world_specs.model_names
        if "target_ball" in model_names:
            return True
        else:
            return False

    def _draw_goal_pos(self):
        del_srv = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
        del_srv.wait_for_service()
        spawn_srv = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        spawn_srv.wait_for_service()

        if self._check_existence:
            del_srv("target_ball")

        spawn_srv(self._create_ball_request(self.goal_pos[0], self.goal_pos[1], self.goal_pos[2]))
        rospy.sleep(1)

    def _init_env_variables(self):
        rospy.logdebug("Init Env Variables...")
        rospy.logdebug("Init Env Variables...END")

    def _set_action(self, action):

        gripper_target = self.get_joint_states()
        gripper_target = gripper_target + action
        gripper_target = np.clip(gripper_target, -math.pi, math.pi)
        self.last_action = "Joint Move"

        self.movement_result = self.move_joints(gripper_target)

        rospy.logwarn("END Set Action ==>" + str(action) +
                      ", NAME=" + str(self.last_action))

    def _get_obs(self):
        current_pose = self.get_ee_pose()
        self._tcp_pose_pub.publish(current_pose)
        obs_pose = np.array([current_pose.position.x,
                             current_pose.position.y,
                             current_pose.position.z,
                             self.goal_pos[0],
                             self.goal_pos[1],
                             self.goal_pos[2], ])

        rospy.logdebug("OBSERVATIONS====>>>>>>>" + str(obs_pose))

        return obs_pose

    def _is_done(self, observations):
        done = np.allclose(a=observations[:3],
                           b=observations[3:],
                           atol=0.05)

        return done or self.no_motion_plan

    def _compute_reward(self, observations, done):
        """
        We punish each step that it passes without achieveing the goal.
        Punishes differently if it reached a position that is imposible to move to.
        Rewards getting to a position close to the goal.
        """

        distance = np.linalg.norm(np.array(observations[:3]) - np.array(observations[3:]))
        ratio = distance / self.init_distance
        reward = -np.ones(1) * ratio
        reward = reward - 10e-2

        if done:
            if self.no_motion_plan:
                reward += -(self.reached_goal_reward * 4)
            else:
                reward += self.reached_goal_reward

        rospy.logwarn(">>>REWARD>>>" + str(reward))

        return reward
