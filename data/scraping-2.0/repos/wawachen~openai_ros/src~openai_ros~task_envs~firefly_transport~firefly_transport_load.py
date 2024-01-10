# from openai_ros.openai_ros.src.openai_ros.task_envs.task_commons import figure8_trajectory_3d
import rospy
import numpy
from gym import spaces
from openai_ros.robot_envs import firefly_env
from gym.envs.registration import register
from geometry_msgs.msg import Point
from geometry_msgs.msg import Vector3
from tf.transformations import euler_from_quaternion
from openai_ros.task_envs.task_commons import LoadYamlFileParamsTest
from openai_ros.task_envs.task_commons import Trajectory, Metrics, figure8_trajectory,figure8_trajectory_3d,figure8_trajectory_3d_xy
from openai_ros.openai_ros_common import ROSLauncher
import os
from mav_msgs.msg import Actuators
import numpy as np
import torch
import time
from geometry_msgs.msg import PoseStamped,Pose
from visualization_msgs.msg import Marker
from nav_msgs.msg import Path
from scipy.io import savemat
import scipy.io 

class FireflyTransportEnv(firefly_env.FireflyDroneEnv):
    def __init__(self):
        """
        Make fireflys learn how to cooperatively transport a load following a trajectory
        """
        ros_ws_abspath = rospy.get_param("/firefly/ros_ws_abspath", None)
        assert ros_ws_abspath is not None, "You forgot to set ros_ws_abspath in your yaml file of your main RL script. Set ros_ws_abspath: \'YOUR/SIM_WS/PATH\'"
        assert os.path.exists(ros_ws_abspath), "The Simulation ROS Workspace path " + ros_ws_abspath + \
                                               " DOESNT exist, execute: mkdir -p " + ros_ws_abspath + \
                                               "/src;cd " + ros_ws_abspath + ";catkin_make"

        ROSLauncher(rospackage_name="rotors_gazebo",
                    launch_file_name="mav_with_waypoint_publisher.launch",
                    ros_ws_abspath=ros_ws_abspath)

        # Load Params from the desired Yaml file
        LoadYamlFileParamsTest(rospackage_name="openai_ros",
                               rel_path_from_package_to_file="src/openai_ros/task_envs/firefly_transport/config",
                               yaml_file_name="firefly_transport_with_two.yaml")

        # Only variable needed to be set here
        number_actions = rospy.get_param('/firefly/n_actions')
        # self.space_3d = rospy.get_param("/firefly/3d_space")
        # high_act = numpy.array([0.5, 1.0])
        # low_act = numpy.array([-0.5, -1.0])
        
        high_act = numpy.array([0.7, 0.7, 0.58])
        low_act = numpy.array([-0.7, -0.7, -0.58])
        # high_act = numpy.array([0.8, 0.8, 1.0])
        # low_act = numpy.array([-0.8, -0.8, -1.0])

        self.action_space = spaces.Box(low=low_act, high=high_act, dtype=np.float32)
        self.action_dim = self.action_space.shape[0]

        # We set the reward range, which is not compulsory but here we do it.
        self.reward_range = (-numpy.inf, numpy.inf)

        # Get WorkSpace Cube Dimensions
        self.work_space_x_max = rospy.get_param("/firefly/work_space/x_max")
        self.work_space_x_min = rospy.get_param("/firefly/work_space/x_min")
        self.work_space_y_max = rospy.get_param("/firefly/work_space/y_max")
        self.work_space_y_min = rospy.get_param("/firefly/work_space/y_min")
        self.work_space_z_max = rospy.get_param("/firefly/work_space/z_max")
        self.work_space_z_min = rospy.get_param("/firefly/work_space/z_min")
        
        self.horizon = rospy.get_param("/firefly/plan_hor")
        self.to_n = rospy.get_param("/firefly/npart")*rospy.get_param("/firefly/popsize")
        self.visualization = rospy.get_param("/firefly/visualization")

        # Maximum RPY values
        self.max_roll = rospy.get_param("/firefly/max_roll")
        self.max_pitch = rospy.get_param("/firefly/max_pitch")
        self.max_yaw = rospy.get_param("/firefly/max_yaw")

        # Get route
        wrap_func = lambda p: self._get_cost_3d(p)

        self.waypoints = None
        self.metric = None
        self.waiting_for_next = None
        self.loop1 = False
        self.np_weights = np.array([100,100])
        self.np_weights1 = np.array([100,100,100])
        self.cost_tol = np.array([0.02,0.02]).dot(self.np_weights) #[0.01, 0.01]
        self.cost_tol1 = np.array([0.02,0.02,0.02]).dot(self.np_weights1) #[0.01, 0.01]
        self._curr_goal_pos = None
        # self._initial_goal_pos = [1.0,0.0,1.4]
        self.has_obstacle = 0

        if self.visualization:
            self.path_pub = rospy.Publisher('desire_path', Path, latch=True, queue_size=10)

            self.fence_pub1 = rospy.Publisher('fence1', Path, latch=True, queue_size=10)
            self.fence_pub2 = rospy.Publisher('fence2', Path, latch=True, queue_size=10)
            self.fence_pub3 = rospy.Publisher('fence3', Path, latch=True, queue_size=10)
            self.fence_pub4 = rospy.Publisher('fence4', Path, latch=True, queue_size=10)
            path_fence1 = Path()
            path_fence2 = Path()
            path_fence3 = Path()
            path_fence4 = Path()
            
            path = Path()

            fence_points1 = np.array([[0.0,-2.0,0.0],[0.0,-2.0,2.0],[4.0,-2.0,2.0],[4.0,-2.0,0.0],[0.0,-2.0,0.0]])
            fence_points2 = np.array([[0.0,2.0,0.0],[0.0,2.0,2.0],[0.0,-2.0,2.0],[0.0,-2.0,0.0],[0.0,2.0,0.0]])
            fence_points3 = np.array([[0.0,2.0,0.0],[0.0,2.0,2.0],[4.0,2.0,2.0],[4.0,2.0,0.0],[0.0,2.0,0.0]])
            fence_points4 = np.array([[4.0,2.0,0.0],[4.0,2.0,2.0],[4.0,-2.0,2.0],[4.0,-2.0,0.0],[4.0,2.0,0.0]])
            # fence_points = np.array([[0.31,0.0,1.86],[3.63,0.0,1.86],[3.63,0.0,0.0],[0.31,0.0,0.0],[0.31,0.0,1.86]])
            for i in range(fence_points1.shape[0]):
                fence_pose_stamped1 = PoseStamped()
                fence_pose_stamped1.pose.position.x = fence_points1[i,0]
                fence_pose_stamped1.pose.position.y = fence_points1[i,1]
                fence_pose_stamped1.pose.position.z = fence_points1[i,2]

                fence_pose_stamped1.header.stamp = rospy.get_rostime()
                fence_pose_stamped1.header.frame_id = "world"

                path_fence1.poses.append(fence_pose_stamped1)

            for i in range(fence_points2.shape[0]):
                fence_pose_stamped2 = PoseStamped()
                fence_pose_stamped2.pose.position.x = fence_points2[i,0]
                fence_pose_stamped2.pose.position.y = fence_points2[i,1]
                fence_pose_stamped2.pose.position.z = fence_points2[i,2]

                fence_pose_stamped2.header.stamp = rospy.get_rostime()
                fence_pose_stamped2.header.frame_id = "world"

                path_fence2.poses.append(fence_pose_stamped2)

            for i in range(fence_points3.shape[0]):
                fence_pose_stamped3 = PoseStamped()
                fence_pose_stamped3.pose.position.x = fence_points3[i,0]
                fence_pose_stamped3.pose.position.y = fence_points3[i,1]
                fence_pose_stamped3.pose.position.z = fence_points3[i,2]

                fence_pose_stamped3.header.stamp = rospy.get_rostime()
                fence_pose_stamped3.header.frame_id = "world"

                path_fence3.poses.append(fence_pose_stamped3)

            for i in range(fence_points4.shape[0]):
                fence_pose_stamped4 = PoseStamped()
                fence_pose_stamped4.pose.position.x = fence_points4[i,0]
                fence_pose_stamped4.pose.position.y = fence_points4[i,1]
                fence_pose_stamped4.pose.position.z = fence_points4[i,2]

                fence_pose_stamped4.header.stamp = rospy.get_rostime()
                fence_pose_stamped4.header.frame_id = "world"

                path_fence4.poses.append(fence_pose_stamped4)

            path_fence1.header.frame_id = "world"
            path_fence1.header.stamp = rospy.get_rostime()
            self.fence_pub1.publish(path_fence1)

            path_fence2.header.frame_id = "world"
            path_fence2.header.stamp = rospy.get_rostime()
            self.fence_pub2.publish(path_fence2)

            path_fence3.header.frame_id = "world"
            path_fence3.header.stamp = rospy.get_rostime()
            self.fence_pub3.publish(path_fence3)

            path_fence4.header.frame_id = "world"
            path_fence4.header.stamp = rospy.get_rostime()
            self.fence_pub4.publish(path_fence4)

        if rospy.get_param("/firefly/route") == "figure8":
            #figure8_trajectory(2.0, 1.0, 0.35,num_points_per_rot=100) points less speed fast
            self.waypoints = figure8_trajectory_3d(2.0, 1.0, 0.35)
            # self.waypoints = figure8_trajectory_3d_xy(2.0, 0.0, 0.35)

            if self.visualization:
                for waypoint in self.waypoints:
                    path_pose_stamped = PoseStamped()
                    
                    path_pose_stamped.pose.position.x = waypoint[0]
                    path_pose_stamped.pose.position.y = waypoint[1]
                    path_pose_stamped.pose.position.z = waypoint[2]

                    path_pose_stamped.header.stamp = rospy.get_rostime()
                    path_pose_stamped.header.frame_id = "world"

                    path.poses.append(path_pose_stamped)

            self.metric = Metrics.wait_start_sequential(Metrics.ext_function_thresh(self.cost_tol1, wrap_func))
            self.reward_mask = lambda pos, i: self._get_cost_reward_3d(pos) if i > 1 else 0
            self.waiting_for_next = lambda i: False if i > 1 else True  # waits only for first point

        if rospy.get_param("/firefly/route") == "figure8_1":
            self.waypoints = figure8_trajectory_3d_xy(2.0, 0.0, 0.35,num_points_per_rot=100)

            if self.visualization:
                for waypoint in self.waypoints:
                    path_pose_stamped = PoseStamped()
                    
                    path_pose_stamped.pose.position.x = waypoint[0]
                    path_pose_stamped.pose.position.y = waypoint[1]
                    path_pose_stamped.pose.position.z = waypoint[2]

                    path_pose_stamped.header.stamp = rospy.get_rostime()
                    path_pose_stamped.header.frame_id = "world"

                    path.poses.append(path_pose_stamped)

            
            self.metric = Metrics.wait_start_sequential(Metrics.ext_function_thresh(self.cost_tol1, wrap_func))
            self.reward_mask = lambda pos, i: self._get_cost_reward_3d(pos) if i >= 0 else 0
            self.waiting_for_next = lambda i: False if i >= 0 else True  # waits only for first point

        if rospy.get_param("/firefly/route") == "square_xy":
            self.obs_pub_makers = rospy.Publisher('/obs_makers', Marker, queue_size=10)
            obs_mat = scipy.io.loadmat("/home/wawa/catkin_meta/src/MBRL_transport/obs_points.mat")
            self.has_obstacle = 0
            self.obs_p = obs_mat["obs"]
            self.obs_p1 = obs_mat["obs1"]
            self.obs_p_pos = obs_mat["obs_pos"] 
            self.obs_p1_pos = obs_mat["obs_pos1"]

            #transform into 256x256 grid
            d_t = 0.02
            len_x = 2
            len_y = 2

            len_y1 = 0.1
            len_x1 = 0.4
            len_y2 = 0.8
            len_x2 = 0.4
            len_y3 = 1.1

            len_x3 = 1.5
            len_y4 = 0.5
            len_x4 = 0.5
            len_y5 = 1.5

            num_pts_per_side_x = round(len_x/d_t)  # 30 seconds
            num_pts_per_side_y = round(len_y/d_t)  # 15 seconds

            num_pts_per_side_y1 = round(len_y1/d_t)
            num_pts_per_side_x1 = round(len_x1/d_t)
            num_pts_per_side_y2 = round(len_y2/d_t)
            num_pts_per_side_x2 = round(len_x2/d_t)
            num_pts_per_side_y3 = round(len_y3/d_t)

            num_pts_per_side_x3 = round(len_x3/d_t)
            num_pts_per_side_y4 = round(len_y4/d_t)
            num_pts_per_side_x4 = round(len_x4/d_t)
            num_pts_per_side_y5 = round(len_y5/d_t)

            center = np.array([2.0, 0.0, 0.8])
            inc_x = len_x / num_pts_per_side_x
            inc_y = len_y / num_pts_per_side_y

            inc_x1 = len_x1 / num_pts_per_side_x1
            inc_y1 = len_y1 / num_pts_per_side_y1
            inc_x2 = len_x2 / num_pts_per_side_x2
            inc_y2 = len_y2 / num_pts_per_side_y2
            inc_x3 = len_x3 / num_pts_per_side_x3
            inc_y3 = len_y3 / num_pts_per_side_y3
            inc_x4 = len_x4 / num_pts_per_side_x4
            inc_y4 = len_y4 / num_pts_per_side_y4
            inc_y5 = len_y5 / num_pts_per_side_y5

            #change here if you have three actions
            self.waypoints = [center - np.array([len_x / 2.0, -len_y / 2.0, 0.0])]  # start
            
            for i in range(num_pts_per_side_x):
                path_component = self.waypoints[-1] + np.array([inc_x, 0, 0])
                self.waypoints += [path_component]
                if self.visualization:
                    path_pose_stamped = PoseStamped()
                    path_pose_stamped.pose.position.x = path_component[0]
                    path_pose_stamped.pose.position.y = path_component[1]
                    path_pose_stamped.pose.position.z = path_component[2]

                    path_pose_stamped.header.stamp = rospy.get_rostime()
                    path_pose_stamped.header.frame_id = "world"

                    path.poses.append(path_pose_stamped)

            for i in range(num_pts_per_side_y1):
                path_component1 = self.waypoints[-1] + np.array([0, -inc_y1, 0])
                self.waypoints += [path_component1]
                if self.visualization:
                    path_pose_stamped = PoseStamped()
                    path_pose_stamped.pose.position.x = path_component1[0]
                    path_pose_stamped.pose.position.y = path_component1[1]
                    path_pose_stamped.pose.position.z = path_component1[2]

                    path_pose_stamped.header.stamp = rospy.get_rostime()
                    path_pose_stamped.header.frame_id = "world"

                    path.poses.append(path_pose_stamped)

            for i in range(num_pts_per_side_x1):
                path_component2 = self.waypoints[-1] + np.array([-inc_x1, 0, 0])
                self.waypoints += [path_component2]
                if self.visualization:
                    path_pose_stamped = PoseStamped()
                    path_pose_stamped.pose.position.x = path_component2[0]
                    path_pose_stamped.pose.position.y = path_component2[1]
                    path_pose_stamped.pose.position.z = path_component2[2]

                    path_pose_stamped.header.stamp = rospy.get_rostime()
                    path_pose_stamped.header.frame_id = "world"

                    path.poses.append(path_pose_stamped)

            for i in range(num_pts_per_side_y2):
                path_component3 = self.waypoints[-1] + np.array([0, -inc_y2, 0])
                self.waypoints += [path_component3]
                if self.visualization:
                    path_pose_stamped = PoseStamped()
                    path_pose_stamped.pose.position.x = path_component3[0]
                    path_pose_stamped.pose.position.y = path_component3[1]
                    path_pose_stamped.pose.position.z = path_component3[2]

                    path_pose_stamped.header.stamp = rospy.get_rostime()
                    path_pose_stamped.header.frame_id = "world"

                    path.poses.append(path_pose_stamped)

            for i in range(num_pts_per_side_x2):
                path_component4 = self.waypoints[-1] + np.array([inc_x2, 0, 0])
                self.waypoints += [path_component4]
                if self.visualization:
                    path_pose_stamped = PoseStamped()
                    path_pose_stamped.pose.position.x = path_component4[0]
                    path_pose_stamped.pose.position.y = path_component4[1]
                    path_pose_stamped.pose.position.z = path_component4[2]

                    path_pose_stamped.header.stamp = rospy.get_rostime()
                    path_pose_stamped.header.frame_id = "world"

                    path.poses.append(path_pose_stamped)
            
            for i in range(num_pts_per_side_y3):
                path_component5 = self.waypoints[-1] + np.array([0, -inc_y3, 0])
                self.waypoints += [path_component5]
                if self.visualization:
                    path_pose_stamped = PoseStamped()
                    path_pose_stamped.pose.position.x = path_component5[0]
                    path_pose_stamped.pose.position.y = path_component5[1]
                    path_pose_stamped.pose.position.z = path_component5[2]

                    path_pose_stamped.header.stamp = rospy.get_rostime()
                    path_pose_stamped.header.frame_id = "world"

                    path.poses.append(path_pose_stamped)

            for i in range(num_pts_per_side_x3):
                path_component6 = self.waypoints[-1] + np.array([-inc_x3, 0, 0])
                self.waypoints += [path_component6]
                if self.visualization:
                    path_pose_stamped = PoseStamped()
                    path_pose_stamped.pose.position.x = path_component6[0]
                    path_pose_stamped.pose.position.y = path_component6[1]
                    path_pose_stamped.pose.position.z = path_component6[2]

                    path_pose_stamped.header.stamp = rospy.get_rostime()
                    path_pose_stamped.header.frame_id = "world"

                    path.poses.append(path_pose_stamped)

            for i in range(num_pts_per_side_y4):
                path_component7 = self.waypoints[-1] + np.array([0, inc_y4, 0])
                self.waypoints += [path_component7]
                if self.visualization:
                    path_pose_stamped = PoseStamped()
                    path_pose_stamped.pose.position.x = path_component7[0]
                    path_pose_stamped.pose.position.y = path_component7[1]
                    path_pose_stamped.pose.position.z = path_component7[2]

                    path_pose_stamped.header.stamp = rospy.get_rostime()
                    path_pose_stamped.header.frame_id = "world"

                    path.poses.append(path_pose_stamped)

            for i in range(num_pts_per_side_x4):
                path_component8 = self.waypoints[-1] + np.array([-inc_x4, 0, 0])
                self.waypoints += [path_component8]
                if self.visualization:
                    path_pose_stamped = PoseStamped()
                    path_pose_stamped.pose.position.x = path_component8[0]
                    path_pose_stamped.pose.position.y = path_component8[1]
                    path_pose_stamped.pose.position.z = path_component8[2]

                    path_pose_stamped.header.stamp = rospy.get_rostime()
                    path_pose_stamped.header.frame_id = "world"

                    path.poses.append(path_pose_stamped)

            for i in range(num_pts_per_side_y5):
                path_component9 = self.waypoints[-1] + np.array([0, inc_y5, 0])
                self.waypoints += [path_component9]
                if self.visualization:
                    path_pose_stamped = PoseStamped()
                    path_pose_stamped.pose.position.x = path_component9[0]
                    path_pose_stamped.pose.position.y = path_component9[1]
                    path_pose_stamped.pose.position.z = path_component9[2]

                    path_pose_stamped.header.stamp = rospy.get_rostime()
                    path_pose_stamped.header.frame_id = "world"

                    path.poses.append(path_pose_stamped)
            
            
            self.metric = Metrics.wait_start_sequential(Metrics.ext_function_thresh(self.cost_tol1, wrap_func))
            self.reward_mask = lambda pos, i: self._get_cost_reward_3d(pos) if i >=0 else 0
            self.waiting_for_next = lambda i: False if i >= 0 else True  # waits only for first point

            ######################################/home/wawa/catkin_meta/src/MBRL_transport/corrective_path_obs/cross/task3/10/save_corrective_waypoints_collision_cross_2.mat
            waypoints_mat = scipy.io.loadmat("/home/wawa/catkin_meta/src/MBRL_transport/corrective_path_obs/square/task3/3/2d_save_corrective_waypoints_collision_square_2.mat")
            # /home/wawa/catkin_meta/src/MBRL_transport/corrective_path_obs/square/task1/1/2d_save_corrective_waypoints_collision_square_0.mat
            # waypoints_mat = scipy.io.loadmat("/home/wawa/catkin_meta/src/MBRL_transport/save_corrective_waypoints_collision_cross_1.mat")
            waypoints_mat_np = waypoints_mat['load']
            self.waypoints = []
            for i in range(waypoints_mat_np.shape[0]):
                self.waypoints.append(waypoints_mat_np[i,:])


            #######################################

        if rospy.get_param("/firefly/route") == "cross":
            self.obs_pub_makers = rospy.Publisher('/obs_makers', Marker, queue_size=10)
            # obs_mat = scipy.io.loadmat("/home/wawa/catkin_meta/src/MBRL_transport/obs_points.mat")
            # self.has_obstacle = 0
            # self.obs_p = obs_mat["obs"]
            # self.obs_p1 = obs_mat["obs1"]
            # self.obs_p_pos = obs_mat["obs_pos"] 
            # self.obs_p1_pos = obs_mat["obs_pos1"]

            #transform into 256x256 grid
            waypoints_mat1 = scipy.io.loadmat("/home/wawa/catkin_meta/src/MBRL_transport/save_waypoints_collision_cross_0.mat")
            waypoints_mat_np1 = waypoints_mat1['load']
    
            for i in range(waypoints_mat_np1.shape[0]):
                if self.visualization:
                    path_pose_stamped = PoseStamped()
                    path_pose_stamped.pose.position.x = waypoints_mat_np1[i,0]
                    path_pose_stamped.pose.position.y = waypoints_mat_np1[i,1]
                    path_pose_stamped.pose.position.z = waypoints_mat_np1[i,2]

                    path_pose_stamped.header.stamp = rospy.get_rostime()
                    path_pose_stamped.header.frame_id = "world"

                    path.poses.append(path_pose_stamped)

            self.metric = Metrics.wait_start_sequential(Metrics.ext_function_thresh(self.cost_tol1, wrap_func))
            self.reward_mask = lambda pos, i: self._get_cost_reward_3d(pos) if i >=0 else 0
            self.waiting_for_next = lambda i: False if i >= 0 else True  # waits only for first point

            ######################################
            waypoints_mat = scipy.io.loadmat("/home/wawa/catkin_meta/src/MBRL_transport/corrective_path_obs/cross/task2/3/save_corrective_waypoints_collision_cross_1.mat")
            # /home/wawa/catkin_meta/src/MBRL_transport/corrective_path_obs/cross/task1/1/save_corrective_waypoints_collision_cross_2.mat
            waypoints_mat_np = waypoints_mat['load']
            self.waypoints = []
            for i in range(waypoints_mat_np.shape[0]):
                self.waypoints.append(waypoints_mat_np[i,:])


        if rospy.get_param("/firefly/route") == "square":
            
            # iterates along sides
            d_t = 0.02
            len_x = 2
            len_z = 0.8

            num_pts_per_side_x = round(len_x/d_t)  # 30 seconds
            num_pts_per_side_z = round(len_z/d_t)  # 30 seconds
            num_pts_per_side_x1 = round(len_x/d_t) # 30 seconds
            num_pts_per_side_z1 = round(len_z/d_t)  # 15 seconds

            center = np.array([2.0, 0.0, 0.8])
            inc_x = len_x / num_pts_per_side_x
            inc_z = len_z / num_pts_per_side_z
            inc_x1 = len_x / num_pts_per_side_x1
            inc_z1 = len_z / num_pts_per_side_z1
            #change here if you have three actions
            self.waypoints = [center - np.array([len_x / 2.0, 0.0, -len_z / 2.0])]  # start

            # clockwise
            for i in range(num_pts_per_side_x):
                path_component = self.waypoints[-1] + np.array([inc_x, 0, 0])
                self.waypoints += [path_component]
                if self.visualization:
                    path_pose_stamped = PoseStamped()
                    path_pose_stamped.pose.position.x = path_component[0]
                    path_pose_stamped.pose.position.y = path_component[1]
                    path_pose_stamped.pose.position.z = path_component[2]

                    path_pose_stamped.header.stamp = rospy.get_rostime()
                    path_pose_stamped.header.frame_id = "world"

                    path.poses.append(path_pose_stamped)

            for i in range(num_pts_per_side_z):
                path_component1 = self.waypoints[-1] + np.array([0,0, -inc_z])
                self.waypoints += [path_component1]
                if self.visualization:
                    path_pose_stamped = PoseStamped()
                    path_pose_stamped.pose.position.x = path_component1[0]
                    path_pose_stamped.pose.position.y = path_component1[1]
                    path_pose_stamped.pose.position.z = path_component1[2]

                    path_pose_stamped.header.stamp = rospy.get_rostime()
                    path_pose_stamped.header.frame_id = "world"

                    path.poses.append(path_pose_stamped)
                   
            for i in range(num_pts_per_side_x1):
                path_component2 = self.waypoints[-1] + np.array([-inc_x1, 0, 0])
                self.waypoints += [path_component2]
                if self.visualization:
                    path_pose_stamped = PoseStamped()
                    path_pose_stamped.pose.position.x = path_component2[0]
                    path_pose_stamped.pose.position.y = path_component2[1]
                    path_pose_stamped.pose.position.z = path_component2[2]

                    path_pose_stamped.header.stamp = rospy.get_rostime()
                    path_pose_stamped.header.frame_id = "world"

                    path.poses.append(path_pose_stamped)

            for i in range(num_pts_per_side_z1):
                path_component5 = self.waypoints[-1] + np.array([0, 0, inc_z1])
                self.waypoints += [path_component5]
                if self.visualization:
                    path_pose_stamped = PoseStamped()
                    path_pose_stamped.pose.position.x = path_component5[0]
                    path_pose_stamped.pose.position.y = path_component5[1]
                    path_pose_stamped.pose.position.z = path_component5[2]

                    path_pose_stamped.header.stamp = rospy.get_rostime()
                    path_pose_stamped.header.frame_id = "world"

                    path.poses.append(path_pose_stamped)
            
            self.metric = Metrics.wait_start_sequential(Metrics.ext_function_thresh(self.cost_tol1, wrap_func))
            self.reward_mask = lambda pos, i: self._get_cost_reward_3d(pos) if i >= 0 else 0
            self.waiting_for_next = lambda i: False if i >= 0 else True  # waits only for first point

        if self.visualization:
            path.header.frame_id = "world"
            path.header.stamp = rospy.get_rostime()
            self.path_pub.publish(path)
                
        savemat('/home/wawa/catkin_meta/src/MBRL_transport/current_waypoints.mat', mdict={'arr': self.waypoints})
        self.trajectory = Trajectory(self.waypoints,
                                     self.metric,  # moves on when func(obs, goal) < thresh
                                     self.waiting_for_next,
                                     loop=self.loop1)

        high = numpy.array([self.work_space_x_max,
                            self.work_space_y_max,
                            self.work_space_z_max,
                            self.work_space_x_max,
                            self.work_space_y_max,
                            self.work_space_z_max,
                            self.work_space_x_max,
                            self.work_space_y_max,
                            self.work_space_z_max,
                            self.max_pitch,
                            self.max_roll,
                            self.max_yaw])

        low = numpy.array([self.work_space_x_min,
                        self.work_space_y_min,
                        self.work_space_z_min,
                        self.work_space_x_min,
                        self.work_space_y_min,
                        self.work_space_z_min,
                        self.work_space_x_min,
                        self.work_space_y_min,
                        self.work_space_z_min,
                        -1*self.max_pitch,
                        -1*self.max_roll,
                        -1*self.max_yaw
                        ])

        self.observation_space = spaces.Box(low, high)

        rospy.logdebug("ACTION SPACES TYPE===>"+str(self.action_space))
        rospy.logdebug("OBSERVATION SPACES TYPE===>" +
                       str(self.observation_space))

        self.cumulated_steps = 0.0
        self.max_x = 4.0
        self.max_y = 2.0
        self.max_z = 2.0
        self.xy = rospy.get_param("/firefly/xy")

        # Here we will add any init functions prior to starting the MyRobotEnv
        super(FireflyTransportEnv, self).__init__(ros_ws_abspath)
        # self.reset_model()

    def reset_drones(self, L):
        bar_odm = self.get_bar_odometry()

        # We get the orientation of the cube in RPY
        b_pos = bar_odm.pose.pose.position
        # L = 0.6

        goal = np.zeros(3)
        goal[0] = b_pos.x
        
        goal[1] = b_pos.y

        goal[2] = b_pos.z+0.35
        
        pose1 = PoseStamped()
        pose1.header.stamp = rospy.Time.now()
        pose1.pose = Pose()
        pose1.pose.position.x = goal[0]+L/2 
        pose1.pose.position.y = goal[1]
        pose1.pose.position.z = goal[2]
        pose1.pose.orientation.w = 0.0

        # create PoseStamped
        pose2 = PoseStamped()
        pose2.header.stamp = rospy.Time.now()
        pose2.pose = Pose()
        pose2.pose.position.x = goal[0]-L/2 
        pose2.pose.position.y = goal[1]
        pose2.pose.position.z = goal[2]
        pose2.pose.orientation.w = 0.0
        

        self.gazebo.unpauseSim()

		# send PoseStamped
        self._cmd_pos_pub1.publish(pose1)
        self._cmd_pos_pub2.publish(pose2)
        time.sleep(5)

        self.gazebo.pauseSim()
        

    def reset_model(self):
        self.trajectory.reset()
        obs, _ = self._get_obs()
        self.set_target(self.trajectory.next(obs[[6,7,8]]))

        return obs

    # def _set_init_pose(self):
    #     """
    #     Sets the Robot in its init linear and angular speeds
    #     and lands the robot. Its preparing it to be reseted in the world.
    #     """
    #     #raw_input("INIT SPEED PRESS")
    #     self.move_pos_base([0.0,0.0,0.0])
    #     # We Issue the landing command to be sure it starts landing
    #     #raw_input("LAND PRESS")
    #     # self.land()

    #     return True        

    def _init_env_variables(self,L):
        """
        Inits variables needed to be initialised each time we reset at the start
        of an episode.
        :return:
        """
        #raw_input("TakeOFF PRESS")
        # We TakeOff before sending any movement commands
        self.takeoff(L)

        # For Info Purposes
        self.cumulated_reward = 0.0
        # # We get the initial pose to mesure the distance from the desired point.
        # gt_pose1 = self.get_odometry1().pose.pose
        # gt_pose2 = self.get_odometry2().pose.pose
        # # self.previous_distance_from_des_point = self.get_distance_from_desired_point(
        # #     gt_pose.position)

    def move_pos(self, L):
        """
        This function is defined for collecting data of dynamics model
        goal:[x,y,z]
        return normlize reference goal :x,z
        we can use this to deduce [dx,dz] position 
        """

        goal = [0.0,0.0,0.0]
        goal[0] = self.goal_joy[0]
        goal[1] = self.goal_joy[1]
        goal[2] = self.goal_joy[2]

        # L = 0.6

        self.pub_action_goal_collection(goal)

        step = 1

        # actions = []
        # obs = []
        # obs1 = []
        # obs2 = []

        #######################################
        # create PoseStamped
        pose1 = PoseStamped()
        pose1.header.stamp = rospy.Time.now()
        pose1.pose = Pose()
        pose1.pose.position.x = goal[0]+L/2 
        pose1.pose.position.y = goal[1]
        pose1.pose.position.z = goal[2]
        pose1.pose.orientation.w = 0.0

        # create PoseStamped
        pose2 = PoseStamped()
        pose2.header.stamp = rospy.Time.now()
        pose2.pose = Pose()
        pose2.pose.position.x = goal[0]-L/2 
        pose2.pose.position.y = goal[1]
        pose2.pose.position.z = goal[2]
        pose2.pose.orientation.w = 0.0

        self.gazebo.unpauseSim()

		# send PoseStamped
        self._cmd_pos_pub1.publish(pose1)
        self._cmd_pos_pub2.publish(pose2)
        ######################################
        obs1 = self._get_uav_obs()[0]
        obs2 = self._get_uav_obs()[1]

        actions1 = np.array([goal[0]/self.max_x,goal[1]/self.max_y,goal[2]/self.max_z])
        centre_x = (obs1[0]+obs2[0])/2.0
        centre_y = (obs1[1]+obs2[1])/2.0
        centre_z = (obs1[2]+obs2[2])/2.0
        actions = np.array([(goal[0]-centre_x), (goal[1]-centre_y), (goal[2]-centre_z)])

        step+=1
        # rate.sleep()
        time.sleep(0.15)
        self.gazebo.pauseSim()
        obs = self._get_obs()[0]
        
        return actions,actions1,obs,obs1,obs2 

    def _set_pos(self, L):
        acts,acts1,obs,obs1,obs2 = self.move_pos(L)
        return acts,acts1,obs,obs1,obs2

    def _set_pos_replay(self, waypoint, L):
        acts,obs,obs1,obs2 = self.move_pos_replay(waypoint,L)
        return acts,obs,obs1,obs2

    def move_pos_replay(self, waypoint,L):
        """
        This function is defined for collecting data of dynamics model
        goal:[x,y,z,yaw,delay]
        action:[dx,dy,dz] position deviation within [-1,1] according to the centre position  
        """

        goal = [0.0,0.0,0.0]
        
        goal[0] = waypoint[0]
        goal[1] = waypoint[1]
        goal[2] = waypoint[2]

        # L = 0.6

        self.pub_action_goal_collection(goal)

        step = 1

        # actions = []
        # obs = []
        # obs1 = []
        # obs2 = []

        #######################################
        # create PoseStamped
        pose1 = PoseStamped()
        pose1.header.stamp = rospy.Time.now()
        pose1.pose = Pose()
        pose1.pose.position.x = goal[0]+L/2 
        pose1.pose.position.y = goal[1]
        pose1.pose.position.z = goal[2]
        pose1.pose.orientation.w = 0.0

        # create PoseStamped
        pose2 = PoseStamped()
        pose2.header.stamp = rospy.Time.now()
        pose2.pose = Pose()
        pose2.pose.position.x = goal[0]-L/2 
        pose2.pose.position.y = goal[1]
        pose2.pose.position.z = goal[2]
        pose2.pose.orientation.w = 0.0

        self.gazebo.unpauseSim()

		# send PoseStamped
        self._cmd_pos_pub1.publish(pose1)
        self._cmd_pos_pub2.publish(pose2)
        ######################################
        obs1 = self._get_uav_obs()[0]
        obs2 = self._get_uav_obs()[1]

        centre_x = (obs1[0]+obs2[0])/2.0
        centre_y = (obs1[1]+obs2[1])/2.0
        centre_z = (obs1[2]+obs2[2])/2.0
        actions = np.array([(goal[0]-centre_x), (goal[1]-centre_y), (goal[2]-centre_z)])

        step+=1
        # rate.sleep()
        time.sleep(0.15)
        self.gazebo.pauseSim()
        obs = self._get_obs()[0]
        # obs1 = self._get_uav_obs()[0]
        # obs2 = self._get_uav_obs()[1]
            
        return actions,obs,obs1,obs2

    def _set_action(self, action, L):
        """
        This set action will Set the linear and angular speed of the drone
        based on the action number given.
        :param action: The action integer that set s what movement to do next.
        """

        # rospy.logdebug("Start Set Action ==>"+str(action))
        # We convert the actions to speed movements to send to the parent class of Parrot
        # We tell drone the linear and angular speed to set to execute
        if rospy.get_param("/firefly/route") == "square_xy":
            obs = Marker()
            obs.header.stamp = rospy.Time.now()
            obs.header.frame_id = "world"
            obs.type = obs.CYLINDER;  
            obs.action = obs.ADD
            obs.id = 200
            obs.pose.position.x = 3
            obs.pose.position.y = 0.5
            obs.pose.position.z = 0.75
            obs.pose.orientation.w = 1.0
            obs.scale.x = .4
            obs.scale.y = .4
            obs.scale.z = 1.5
            obs.color.a = 1
            obs.color.r = 0.4
            obs.color.g = 0.4
            obs.color.b = 0.2
            obs.lifetime = rospy.Duration()

            obs1 = Marker()
            obs1.header.stamp = rospy.Time.now()
            obs1.header.frame_id = "world"
            obs1.type = obs.CYLINDER;  
            obs1.action = obs.ADD
            obs1.id = 201
            obs1.pose.position.x = 1
            obs1.pose.position.y = -1
            obs1.pose.position.z = 0.75
            obs1.pose.orientation.w = 1.0
            obs1.scale.x = .6   
            obs1.scale.y = .6
            obs1.scale.z = 1.5
            obs1.color.a = 1
            obs1.color.r = 0.4
            obs1.color.g = 0.4
            obs1.color.b = 0.2
            obs1.lifetime = rospy.Duration()

            self.obs_pub_makers.publish(obs)
            self.obs_pub_makers.publish(obs1)
        
        if rospy.get_param("/firefly/route") == "cross":
            obs = Marker()
            obs.header.stamp = rospy.Time.now()
            obs.header.frame_id = "world"
            obs.type = obs.CYLINDER;  
            obs.action = obs.ADD
            obs.id = 200
            obs.pose.position.x = 2.0
            obs.pose.position.y = 0.0
            obs.pose.position.z = 0.75
            obs.pose.orientation.w = 1.0
            obs.scale.x = .4
            obs.scale.y = .4
            obs.scale.z = 1.5
            obs.color.a = 1
            obs.color.r = 0.4
            obs.color.g = 0.4
            obs.color.b = 0.2
            obs.lifetime = rospy.Duration()

            self.obs_pub_makers.publish(obs)


        action_maker = self.move_pos_base(action,L)
        self.pub_action_goal(action_maker)

        # rospy.logdebug("END Set Action ==>"+str(action))

    def _get_uav_obs(self):
        uav1_odo = self.get_odometry1()
        uav2_odo = self.get_odometry2()

        b_pos1 = uav1_odo.pose.pose.position
        b_pos2 = uav2_odo.pose.pose.position

        return np.array([b_pos1.x,b_pos1.y,b_pos1.z]),np.array([b_pos2.x,b_pos2.y,b_pos2.z])
    
    def get_uav_obs(self):
        uav1_odo = self.get_odometry1()
        uav2_odo = self.get_odometry2()

        b_pos1 = uav1_odo.pose.pose.position
        b_pos2 = uav2_odo.pose.pose.position

        return np.array([b_pos1.x,b_pos1.y,b_pos1.z]),np.array([b_pos2.x,b_pos2.y,b_pos2.z])

    def _get_uav_ori(self):
        uav1_odo = self.get_odometry1()
        uav2_odo = self.get_odometry2()

        b_roll1, b_pitch1, b_yaw1 = self.get_orientation_euler(uav1_odo.pose.pose.orientation)
        b_roll2, b_pitch2, b_yaw2 = self.get_orientation_euler(uav2_odo.pose.pose.orientation)

        return np.array([b_roll1, b_pitch1, b_yaw1]),np.array([b_roll2, b_pitch2, b_yaw2])

    def _get_obs(self):
        """
        Here we define what sensor data defines our robots observations
        To know which Variables we have acces to, we need to read the
        droneEnv API DOCS
        :return:
        """
        rospy.logdebug("Start Get Observation ==>")
        # We get the laser scan data
        u1_odm = self.get_odometry1()
        u2_odm = self.get_odometry2()
        bar_odm = self.get_bar_odometry()

        # We get the orientation of the cube in RPY
        b_roll, b_pitch, b_yaw = self.get_orientation_euler(bar_odm.pose.pose.orientation)
        b_pos = bar_odm.pose.pose.position
        uav1_pos = u1_odm.pose.pose.position
        uav2_pos = u2_odm.pose.pose.position
        
        #also track the two drones
        observations = [round(uav1_pos.x,8)/self.max_x,
                            round(uav1_pos.y,8)/self.max_y,
                            round(uav1_pos.z,8)/self.max_z,
                            round(uav2_pos.x,8)/self.max_x,
                            round(uav2_pos.y,8)/self.max_y,
                            round(uav2_pos.z,8)/self.max_z,
                        round(b_pos.x,8)/self.max_x,
                        round(b_pos.y,8)/self.max_y,
                        round(b_pos.z,8)/self.max_z,
                        round(b_roll,8),
                        round(b_pitch,8),
                        round(b_yaw,8)]

        orientations = [b_roll, b_pitch, b_yaw]

        # rospy.logdebug("Observations==>"+str(observations))
        # rospy.logdebug("END Get Observation ==>")
        return np.array(observations), orientations


    def _is_done(self, observations, ob1, ob2, orientations, ori1, ori2):
        """
        obs,obs1,obs2,ori,ori1,ori2
        The done can be done due to three reasons:
        1) It went outside the workspace
        2) It detected something with the sonar that is too close
        3) It flipped due to a crash or something
        4) It has reached the desired point
        """
        episode_done = False
        has_reached_des_point = self.trajectory.is_finished()  # returns true when the full trajectory has been run through

        # for load-UAV system: consider limits of drones, limit of a load 
        current_position = Point()
        
        current_position.x = observations[6]*self.max_x
        current_position.y = observations[7]*self.max_y
        current_position.z = observations[8]*self.max_z

        current_orientation = Point()
        current_orientation.x = orientations[0]
        current_orientation.y = orientations[1]
        current_orientation.z = orientations[2]

        is_inside_workspace_now = self.is_inside_workspace(current_position,ob1,ob2)
        # sonar_detected_something_too_close_now = self.sonar_detected_something_too_close(
        #     sonar_value)
        drone_flipped = self.drone_has_flipped(current_orientation,ori1,ori2)

        rospy.logwarn(">>>>>> DONE RESULTS <<<<<")
        if not is_inside_workspace_now:
            rospy.logerr("is_inside_workspace_now=" +
                         str(is_inside_workspace_now))
        else:
            rospy.logwarn("is_inside_workspace_now=" +
                          str(is_inside_workspace_now))


        if drone_flipped:
            rospy.logerr("drone_flipped="+str(drone_flipped))
        else:
            rospy.logwarn("drone_flipped="+str(drone_flipped))

        # We see if we are outside the Learning Space
        episode_done = not(is_inside_workspace_now) or drone_flipped or has_reached_des_point
        done_f = not(is_inside_workspace_now) or drone_flipped

        if episode_done:
            rospy.logerr("episode_done====>"+str(episode_done))
        else:
            rospy.logwarn("episode_done====>"+str(episode_done))

        return (episode_done,done_f)

    # this weight default comes from the data (normalizing)
    def _get_cost(self, pos):
        # 10 comes from the square term
        assert pos.shape == self._curr_goal_pos.shape
        pos1 = pos.copy()
        pos1[0] = pos1[0]*self.max_x
        pos1[1] = pos1[1]*self.max_z
       
        return np.abs(pos1 - self._curr_goal_pos).dot(self.np_weights)

    def _get_cost_3d(self, pos):
        assert pos.shape[0] == 3
        # 10 comes from the square term
        assert self._curr_goal_pos.shape[0] == 3
        assert pos.shape == self._curr_goal_pos.shape

        pos1 = pos.copy()
        pos1[0] = pos1[0]*self.max_x
        pos1[1] = pos1[1]*self.max_y
        pos1[2] = pos1[2]*self.max_z
       
        return np.abs(pos1 - self._curr_goal_pos).dot(self.np_weights1)
    
    def _get_cost_reward(self, pos):
        # 10 comes from the square term
        assert pos.shape == self._curr_goal_pos.shape
        pos1 = pos.copy()
        pos1[0] = pos1[0]*self.max_x
        pos1[1] = pos1[1]*self.max_z
       
        return np.sqrt(np.sum((pos1 - self._curr_goal_pos)**2))

    def _get_cost_reward_3d(self, pos):
        # 10 comes from the square term
        assert pos.shape == self._curr_goal_pos.shape
        assert pos.shape[0] == 3
        pos1 = pos.copy()
        pos1[0] = pos1[0]*self.max_x
        pos1[1] = pos1[1]*self.max_y
        pos1[2] = pos1[2]*self.max_z
       
        return pos1[0]-self._curr_goal_pos[0], pos1[1]-self._curr_goal_pos[1], pos1[2]-self._curr_goal_pos[2], np.sqrt(np.sum((pos1 - self._curr_goal_pos)**2))


    def set_target(self, xf):
        self._curr_goal_pos = xf


    def is_inside_workspace(self, current_position, obs1, obs2):
        """
        Check if the Drone is inside the Workspace defined
        """
        is_ok_inside = False
        is_inside = False
        is_inside1 = False
        is_inside2 = False

        rospy.logwarn("##### INSIDE WORK SPACE? #######")
        rospy.logwarn("XYZ current_position"+str(current_position))
        rospy.logwarn("work_space_x_max"+str(self.work_space_x_max) +
                      ",work_space_x_min="+str(self.work_space_x_min))
        rospy.logwarn("work_space_y_max"+str(self.work_space_y_max) +
                      ",work_space_y_min="+str(self.work_space_y_min))
        rospy.logwarn("work_space_z_max"+str(self.work_space_z_max) +
                      ",work_space_z_min="+str(self.work_space_z_min))
        rospy.logwarn("############")

        safe_distance_load_z = 0.1
        safe_distance_load_x = 0.5

        if current_position.x > 0.0 and current_position.x < 3.7: #3.34
            if current_position.y > (self.work_space_y_min+0.01) and current_position.y < (self.work_space_y_max-0.01):
                if current_position.z > (self.work_space_z_min+safe_distance_load_z) and current_position.z < (self.work_space_z_max-safe_distance_load_z):
                    is_inside = True

        if obs1[0]>0.31 and obs1[0]<3.63:
            if obs1[1]>(self.work_space_y_min+0.31) and obs1[1]<(self.work_space_y_max-0.31):
                if obs1[2]>0.3 and obs1[2]<1.86:
                    is_inside1 = True

        if obs2[0]>0.31 and obs2[0]<3.63:
            if obs2[1]>(self.work_space_y_min+0.31) and obs2[1]<(self.work_space_y_max-0.31):
                if obs2[2]>0.3 and obs2[2]<1.86:
                    is_inside2 = True

        is_ok_inside = is_inside #and is_inside1 and is_inside2

        return is_ok_inside


    def drone_has_flipped(self, current_orientation, ori1, ori2):
        """
        Based on the orientation RPY given states if the drone has flipped
        """
        has_ok_flipped = True
        has_flipped = True
        has_flipped1 = True
        has_flipped2 = True

        self.max_roll = rospy.get_param("/firefly/max_roll")
        self.max_pitch = rospy.get_param("/firefly/max_pitch")

        rospy.logwarn("#### HAS FLIPPED? ########")
        rospy.logwarn("RPY current_orientation"+str(current_orientation))
        rospy.logwarn("max_roll"+str(self.max_roll) +
                      ",min_roll="+str(-1*self.max_roll))
        rospy.logwarn("max_pitch"+str(self.max_pitch) +
                      ",min_pitch="+str(-1*self.max_pitch))
        rospy.logwarn("############")

        if current_orientation.x > -1*self.max_roll and current_orientation.x <= self.max_roll:
            if current_orientation.y > -1*self.max_pitch and current_orientation.y <= self.max_pitch:
                has_flipped = False
        
        if ori1[0] > -1*self.max_roll and ori1[0] <= self.max_roll:
            if ori1[1] > -1*self.max_pitch and ori1[1] <= self.max_pitch:
                has_flipped1 = False

        if ori2[0] > -1*self.max_roll and ori2[0] <= self.max_roll:
            if ori2[1] > -1*self.max_pitch and ori2[1] <= self.max_pitch:
                has_flipped2 = False

        has_ok_flipped = has_flipped or has_flipped1 or has_flipped2

        return has_ok_flipped


    def get_orientation_euler(self, quaternion_vector):
        # We convert from quaternions to euler
        orientation_list = [quaternion_vector.x,
                            quaternion_vector.y,
                            quaternion_vector.z,
                            quaternion_vector.w]

        roll, pitch, yaw = euler_from_quaternion(orientation_list)
        return roll, pitch, yaw

    def get_goal(self):
        goal = self.get_goal_1()[0, 0]
        next_n = self.trajectory.try_next_n(self.horizon-1)\
            .reshape(self.horizon-1, self.action_dim)
        future_goals = np.concatenate([goal[None], next_n], axis=0)
        future_goals = torch.from_numpy(future_goals).float()
        single_goals = future_goals.detach().clone().cpu().numpy()
        future_goals = future_goals[None]
        trans = future_goals.transpose(0, 1)
        #[H,1,dim1]
        self.pub_desire_goals(trans)
        future_goals = trans.expand(-1, self.to_n, -1)
        assert future_goals.shape[1] == self.to_n
        #[H, nopt * npart, dim1]
        return future_goals,single_goals
    
    def get_goal_1(self):
        #[1,horizon, actdim]
        goal = np.tile(self._curr_goal_pos[None, None], (1, self.horizon, 1))
        return goal

    def pub_desire_goals(self, goals):
        dgs = Marker()
        dgs.header.stamp = rospy.Time.now()
        dgs.header.frame_id = "world"
        dgs.text = "goals" 
        dgs.action = dgs.ADD
        dgs.type = Marker.POINTS;  
        dgs.color.a = 0.5
        dgs.scale.x = 0.1
        dgs.scale.y = 0.1
        dgs.color.g = 1.0
        dgs.pose.orientation.w = 1.0

        for j in range(goals.shape[0]):
            dgs.points.append(Point(goals[j,0,0],goals[j,0,1],goals[j,0,2]))

        self.goal_pub_makers.publish(dgs)

    def pub_corrective_desire_goals(self, goals):
        dgs = Marker()
        dgs.header.stamp = rospy.Time.now()
        dgs.header.frame_id = "world"
        dgs.text = "goals_c" 
        dgs.action = dgs.ADD
        dgs.type = Marker.POINTS;  
        dgs.color.a = 0.5
        dgs.scale.x = 0.1
        dgs.scale.y = 0.1
        dgs.color.r = 0.0
        dgs.color.g = 0.4
        dgs.color.b = 1.0
        dgs.pose.orientation.w = 1.0

        for j in range(goals.shape[0]):
            dgs.points.append(Point(goals[j,0],goals[j,1],goals[j,2]))

        self.goal_pub_makers_c.publish(dgs)

    def pub_action_goal_collection(self, goal):
        dgs = Marker()
        dgs.header.stamp = rospy.Time.now()
        dgs.header.frame_id = "world"
        dgs.text = "action goal" 
        dgs.type = dgs.SPHERE;  
        dgs.action = dgs.ADD

        dgs.pose.position.x = goal[0]
        dgs.pose.position.y = goal[1]
        dgs.pose.position.z = goal[2]
        dgs.pose.orientation.w = 1.0

        dgs.scale.x = .05   
        dgs.scale.y = .05
        dgs.scale.z = .05
        dgs.color.a = 1
        dgs.color.r = 0.0
        dgs.color.g = 0.9
        dgs.color.b = 0.2

        self.action_pub_makers_c.publish(dgs)

    def pub_action_goal(self, goal):
        dgs = Marker()
        dgs.header.stamp = rospy.Time.now()
        dgs.header.frame_id = "world"
        dgs.text = "action goal" 
        dgs.type = dgs.SPHERE;  
        dgs.action = dgs.ADD

        dgs.pose.position.x = goal[0]
        dgs.pose.position.y = goal[1]
        dgs.pose.position.z = goal[2]
        dgs.pose.orientation.w = 1.0

        dgs.scale.x = .05
        dgs.scale.y = .05
        dgs.scale.z = .05
        dgs.color.a = 1
        dgs.color.r = 0.5
        dgs.color.g = 0.9
        dgs.color.b = 0.2

        self.action_pub_makers.publish(dgs)

    def pub_action_sequence(self,s_top):
        # send the best action sequence s_top [T,5,2] x,z needs de-normalization

        for i in range(s_top.shape[1]):
            m = Marker()
            m.header.stamp = rospy.Time.now()
            m.header.frame_id = "world"
            m.id = i+100
            m.type = Marker.LINE_STRIP
            m.lifetime = rospy.Duration()
            m.scale.x = 0.01

            if i == 0:
                m.color.r = 0.0
                m.color.g = 0.0  
                m.color.b = 0.8
                m.color.a = 1.0
            else:
                m.color.r = 0.4
                m.color.g = 0.4  
                m.color.b = 0.4
                m.color.a = 1.0

            # m.pose.position.x = s_top[0,i,0]*self.max_x
            # m.pose.position.y = 0.0
            # m.pose.position.z = s_top[0,i,1]*self.max_z

            m.pose.orientation.w = 1

            for j in range(self.horizon):
                m.points.append(Point(s_top[j,i,0]*self.max_x, s_top[j,i,1]*self.max_y,s_top[j,i,2]*self.max_z))

            self.action_sequence_pub_makers.publish(m)

    def pub_action_sequence1(self,s_bad):
        # send the best action sequence s_top [T,5,2] x,z needs de-normalization

        for i in range(s_bad.shape[1]):
            m = Marker()
            m.header.stamp = rospy.Time.now()
            m.header.frame_id = "world"
            m.id = i+500
            m.type = Marker.LINE_STRIP
            m.lifetime = rospy.Duration()
            m.scale.x = 0.01

            m.color.r = 1.0
            m.color.g = 0.0  
            m.color.b = 0.0
            m.color.a = 1.0

            # m.pose.position.x = s_top[0,i,0]*self.max_x
            # m.pose.position.y = 0.0
            # m.pose.position.z = s_top[0,i,1]*self.max_z

            m.pose.orientation.w = 1

            for j in range(self.horizon):
                m.points.append(Point(s_bad[j,i,0]*self.max_x, s_bad[j,i,1]*self.max_y,s_bad[j,i,2]*self.max_z))

            self.action_sequence_pub_makers1.publish(m)

    # def pub_trajectories(self, full_action, sample_n=10):
    #     acts = full_action.action_sequence.act[0]  # (1, P, horizon, actdim)
    #     order = full_action.results.order[0]  # (1, P) -> (P,)
    #     costs = full_action.results.costs[0]  # (1, P,)
    #     traj = full_action.results.trajectory.obs[0]  # (1, P, H+1, obsdim)

    #     assert acts.shape[0] == order.shape[0] == costs.shape[0] == traj.shape[0]

    #     assert self.horizon == acts.shape[1] == traj.shape[1] - 1

    #     if sample_n == -1:
    #         sample_n = acts.shape[0]
    #     sample_n = min(sample_n, acts.shape[0])
    #     inc = max(1, acts.shape[0] // sample_n)

    #     # for each trajectory, send it as a Marker
    #     for i in range(sample_n):
    #         idx = order[i * inc]
    #         this_as = acts[idx]
    #         this_tr = traj[idx]
    #         this_c = costs[idx]

    #         m = Marker()
    #         m.header.stamp = rospy.Time.now()
    #         m.header.frame_id = "world"
    #         m.ns = 'mpc_vis';  m.id = i;  m.type = Marker.LINE_STRIP
    #         m.lifetime = rospy.Duration()
    #         m.scale.x = 0.05
    #         m.color.r = 0.4;  m.color.g = 0.4;  m.color.b = 0.4;  m.color.a = 1.0

    #         # this is how we pass the cost
    #         m.text = "%f" % this_c

    #         m.pose.position.x = this_tr[0, 0]
    #         m.pose.position.y = this_tr[0, 1]
    #         m.pose.position.z = this_tr[0, 2]

    #         m.pose.orientation.w = 1

    #         for j in range(self.horizon + 1):
    #             if this_tr.shape[1] == 2:
    #                 m.points.append(Point(this_tr[j, 0], this_tr[j, 1], 0.))
    #             else:
    #                 m.points.append(Point(this_tr[j, 0], this_tr[j, 1], this_tr[j, 2]))

    #         self._ros_trajectory_marker_pub.publish(m)

    #     # send the best action sequence too
    #     acm = Marker()
    #     acm.header.stamp = rospy.Time.now()
    #     acm.header.frame_id = "cf"
    #     acm.text = "%f" % costs[order[0]]
    #     acm.type = Marker.LINE_STRIP;  acm.lifetime = rospy.Duration()
    #     for j in range(self.horizon):
    #         ac = acts[order[0], j, :3].tolist()
    #         ac += [0] * (3 - len(ac))  # padding
    #         acm.points.append(Point(*ac))

    #     self._ros_ac_marker_pub.publish(acm)