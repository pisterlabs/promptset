import rospy
import os
import random
import rospkg
import math

import numpy as np
from gym import spaces

from openai_ros.robot_envs import multi_geo_robot_env
from openai_ros.openai_ros_common import ROSLauncher
from gazebo_msgs.srv import SpawnModel, SpawnModelRequest, DeleteModel, GetWorldProperties
from geometry_msgs.msg import Pose, Quaternion, Point

class MultiGeoRobotReachEnv(multi_geo_robot_env.MultiGeoRobotEnv):
    def __init__(self):
        ros_ws_abspath = rospy.get_param("ros_ws_path", None)
        assert ros_ws_abspath is not None, "You forgot to set ros_ws_abspath in your yaml file of your main RL script. Set ros_ws_abspath: \'YOUR/SIM_WS/PATH\'"
        assert os.path.exists(ros_ws_abspath), "The Simulation ROS Workspace path "+ros_ws_abspath + \
            " DOESNT exist, execute: mkdir -p "+ros_ws_abspath + \
            "/src;cd "+ros_ws_abspath+";catkin_make"

        self.launch = ROSLauncher(rospackage_name="multi_geo_robot_desc",
                    launch_file_name="multi_geo_robot_empty.launch",
                    ros_ws_abspath=ros_ws_abspath)

        super().__init__(ros_ws_abspath)

        self.init_pos = np.random.uniform(size=len(self.joints), low=-0, high=0)

        self.n_observations = 6 # EE Robot + Target Point
        self.observation_space = spaces.Box(low=-3.14, high=3.14, shape=(self.n_observations,))
        self.n_actions = len(self.joints)
        self.action_space = spaces.Box(low=-0.1, high=0.1, shape=(self.n_actions,))

        self.reached_goal_reward = 10
        self._set_goal_pos()
        self.init_distance = None

    def _create_ball_request(self, name, px, py, pz):
        target_path = rospkg.RosPack()
        target_path = target_path.get_path('multi_geo_robot_desc')
        target_path += '/models/target/target_point_Blue.urdf'
        target_urdf = None

        with open(target_path, "r") as g:
            target_urdf = g.read()

        target_pose = Pose(Point(x=px,y=py,z=pz), Quaternion(x=0,y=0,z=0,w=1))

        req = SpawnModelRequest()
        req.model_name = name
        req.model_xml = target_urdf
        req.initial_pose = target_pose

        return req

    def _check_existence(self, name):
        get_world_specs = rospy.ServiceProxy('/gazebo/get_world_properties', GetWorldProperties)()
        model_names = get_world_specs.model_names
        if name in model_names:
            return True
        else:
            return False
            
    def _draw_goal_pos(self, name, px, py, pz):
        del_srv = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
        del_srv.wait_for_service()
        spawn_srv = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        spawn_srv.wait_for_service()

        if self._check_existence(name):
            del_srv(name)

        spawn_srv(self._create_ball_request(name, px, py, pz))
        rospy.sleep(1)
        
    def _set_goal_pos(self):
        self.goal_pos = [0.3, 
                        0.3, 
                        0.3,
                        0,0,0,1]
        self._draw_goal_pos("target_ball", self.goal_pos[0],self.goal_pos[1],self.goal_pos[2])
            
    
    # Methods that the TrainingEnvironment will need to define here as virtual
    # because they will be used in RobotGazeboEnv GrandParentClass and defined in the
    # TrainingEnvironment.
    # ----------------------------
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
                    pose.position.z,])

        self.init_distance = np.linalg.norm(np.array(pose[:]) - np.array(self.goal_pos[:3]))

        return True

    def _init_env_variables(self):
        rospy.logdebug("Init Env Variables...")
        rospy.logdebug("Init Env Variables...END")

    def _compute_reward(self, observations, done):
        """
        We punish each step that it passes without achieveing the goal.
        and punish for every movement_result that is False
        Rewards getting to a position close to the goal.
        """
        
        distance = np.linalg.norm(np.array(observations[:3]) - np.array(observations[3:]))
        ratio = distance/self.init_distance
        reward = -np.ones(1)*ratio
        reward = reward - 10e-2

        if done:
            if self.no_motion_plan:
                reward += -(self.reached_goal_reward*4)
            else:
                reward += self.reached_goal_reward
        
        rospy.logwarn(">>>REWARD>>>"+str(reward))

        return reward

    def _set_action(self, action):

        gripper_target = self.get_joint_states()   
        gripper_target = gripper_target + action
        gripper_target = np.clip(gripper_target, -math.pi, math.pi)
        self.last_action = "Joint Move"

        self.movement_result = self.move_joints(gripper_target)
                                           
        rospy.logwarn("END Set Action ==>" + str(action) +
                      "\n==>" + str(self.last_action))

    def _get_obs(self):
        current_pose = self.get_ee_pose()
        obs_pose = np.array([current_pose.position.x,
                    current_pose.position.y,
                    current_pose.position.z,
                    self.goal_pos[0],
                    self.goal_pos[1],
                    self.goal_pos[2],])
                
        rospy.logdebug(("OBSERVATIONS====>>>>>>>\npose:{}").format(obs_pose))
        return obs_pose

    def _is_done(self, observations):
        done = (np.allclose(a=observations[:3], 
                            b=observations[3:], 
                            atol=0.05))

        return done or self.no_motion_plan