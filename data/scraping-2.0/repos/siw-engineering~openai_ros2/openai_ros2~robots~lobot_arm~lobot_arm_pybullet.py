import os
import abc
import threading

from rclpy.qos import qos_profile_services_default, qos_profile_parameters
import rclpy

from openai_ros2.robots.lobot_arm.lobot_arm_base import LobotArmBase
from openai_ros2.sims import Bullet, WorldParams

from ros2_control_interfaces.msg import JointControl

import pybullet as bullet
import pybullet_data

import numpy

from gym.spaces import Box

def spin_node(node):
    rclpy.spin(node)

class LobotArmPybullet(LobotArmBase):
    """Pybullet Lobot Arm"""

    def __init__(self, node):
        super().__init__(node)
        world_params = WorldParams()
        world_params.urdf_path = 'plane.urdf'
        print(f'Current directory: {os.getcwd()}')
        self.bullet = Bullet(node, use_gui=True, robot_urdf_path='/home/dark/biped_ros2/src/openai_ros2/openai_ros2/robots/lobot_arm/lobot_arm_org.urdf', world=world_params)
        joint_control_topic = '/arm_standalone/control'
        self._control_pub = self.node.create_publisher(JointControl, joint_control_topic, qos_profile_parameters)
        self._target_joint_state = numpy.array([0.0, 0.0, 0.0])

        # spin_thread = threading.Thread(target = spin_node, kwargs = {'node': self.node})
        # spin_thread.daemon = True
        # spin_thread.start()



    def set_action(self, action: numpy.ndarray):
        assert len(action) == 3, f'{len(action)} actions passed to LobotArmSim, expected: 3'
        assert action.shape == (3,), f'Expected action shape of {self._target_joint_state.shape}, actual shape: {action.shape}'
        rclpy.spin_once(self.node, timeout_sec=0.1)
        self._target_joint_state += action  # TODO change from += to = and investigate the effects
        self._target_joint_state = self._target_joint_state.clip([-2.356194, -1.570796, -1.570796], [2.356194, 0.500, -1.570796])
        msg = JointControl()
        msg.joints = ['Joint 1', 'Joint 2', 'Joint 3']
        msg.goals = self._target_joint_state.tolist()
        msg.header.stamp = self._current_sim_time.to_msg()
        self._control_pub.publish(msg)
        self.bullet.step()
        rclpy.spin_once(self.node, timeout_sec=0.1)

    def get_action_space(self):
        return Box(-1, 1, shape=(3,))

    def reset(self) -> None:
        self.bullet.reset()
        # TODO: Also do other reset logic here
        pass