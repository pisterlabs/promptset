import copy
import time
from typing import Dict

from gazebo_msgs.msg import ContactsState

import numpy

from openai_ros2.utils import ut_param_server, ut_gazebo
from openai_ros2.utils.gazebo import Gazebo

import rclpy
from rclpy.qos import qos_profile_services_default, qos_profile_parameters
from rclpy.time import Time as rclpyTime

from ros2_control_interfaces.msg import JointControl
from ros2_control_interfaces.srv import GetCurrentSimTime
from sensor_msgs.msg import JointState

from openai_ros2.robots.lobot_arm.lobot_arm_base import LobotArmBase

from gym.spaces import Box


class LobotArmSim(LobotArmBase):
    """Simulated Lobot Arm"""

    '''-------------PUBLIC METHODS START-------------'''

    def __init__(self, node, **kwargs):
        self._gazebo = Gazebo(use_gui=True)
        super().__init__(node, **kwargs)
        self._update_period_ns = 1000000000 / ut_param_server.get_update_rate(self.node)
        # Get robot name from parameter server, this is to ensure that the gazebo plugin subscribing to the control
        # reads the same name as this code, because the topic depends on the robot name
        robot_names = ut_param_server.get_robots(self.node)
        self.robot_name = robot_names[0]
        self._joint_names = ut_param_server.get_joints(self.node, self.robot_name)
        joint_control_topic = '/' + self.robot_name + '/control'
        self._control_pub = self.node.create_publisher(JointControl, joint_control_topic, qos_profile_parameters)
        self._contact_sub = self.node.create_subscription(ContactsState, f'/{self.robot_name}/contacts',
                                                          self.__contact_subscription_callback,
                                                          qos_profile=qos_profile_services_default)

        self._latest_contact_msg = None
        self._target_joint_state = numpy.array([0.0, 0.0, 0.0])
        self._previous_update_sim_time = rclpyTime()

    def set_action(self, action: numpy.ndarray) -> None:
        """
        Sets the action, unpauses the simulation and then waits until the update period of openai gym is over.
        The simulation is also expected to pause at the same time.
        This is to create a deterministic time step for the gym environment such that the agent can properly evaluate
        each action
        :param action:
        :return: obs, reward, done, info
        """
        assert len(action) == 3, f'{len(action)} actions passed to LobotArmSim, expected: 3'
        assert action.shape == (3,), f'Expected action shape of {self._target_joint_state.shape}, actual shape: {action.shape}'
        # Note that if we want to do controller action based on current position, we need to use the noisy position rather than actual
        # This is such that it is closer to the real one
        if self._latest_joint_state_msg is not None:
            obs = self.get_observations()
            current_position = obs.position_data
        else:
            current_position = numpy.array([0.0, 0.0, 0.0])
        self._target_joint_state = current_position + action

        # Old action style, adjusts controller goal
        # self._target_joint_state += action

        self._target_joint_state = self._target_joint_state.clip([-2.356194, -1.570796, -1.570796], [2.356194, 0.500, 1.570796])
        msg = JointControl()
        msg.joints = self._joint_names
        msg.goals = self._target_joint_state.tolist()
        msg.header.stamp = self._current_sim_time.to_msg()
        self._control_pub.publish(msg)
        # Assume the simulation is paused due to the gym training plugin when set_action is called
        self._gazebo.unpause_sim()
        self._spin_until_update_period_over()

    def reset(self) -> None:
        self._gazebo.pause_sim()
        self._gazebo.reset_sim()
        if self.random_init_pos:
            positions = ut_gazebo.random_positions(self.node)
            if positions is not None:
                print(f'Random initial positions: {positions}')
                self._target_joint_state = positions
        for i in range(10):
            rclpy.spin_once(self.node, timeout_sec=0.03)
        self._reset_state()
        # No unpause here because it is assumed that the set_action will unpause it

    def get_action_space(self):
        return Box(-1, 1, shape=(3,))

    '''-------------PUBLIC METHODS END-------------'''

    '''-------------PRIVATE METHODS START-------------'''

    def _reset_state(self) -> None:
        super()._reset_state()
        self._latest_contact_msg = None
        # We assume that if we do random initial position, the target joint state will be set after we call the random and the positions are obtained
        # So we only set target_joint_state when is not random_init_pos
        if not self.random_init_pos:
            # print(f'Setting 0 initial positions')
            self._target_joint_state = numpy.array([0.0, 0.0, 0.0])

        self._previous_update_sim_time = rclpyTime()

    def __contact_subscription_callback(self, message: ContactsState) -> None:
        header_time = message.header.stamp.sec * 1000000000 + message.header.stamp.nanosec
        # print(f"[{message.header.stamp.sec}][{message.header.stamp.nanosec}]Contact!!")
        current_sim_time = self._current_sim_time.nanoseconds
        time_diff = header_time - current_sim_time
        if header_time < current_sim_time - 1000000000:
            print(f"Outdated contact message,ignoring, time_diff: {time_diff}")
            return
        elif header_time > current_sim_time + 400000000:
            print(f"Reset detected, ignoring, time_diff: {time_diff}")
            return
        self._latest_contact_msg = message

    def _spin_until_update_period_over(self) -> None:
        # Loop to block such that when we take observation it is the latest observation when the
        # simulation is paused due to the gym training plugin
        # Also have a timeout such that when the loop gets stuck it will break itself out
        timeout_duration = 1.5
        loop_start_time = time.time()
        while True:
            # spinning the node will cause self._current_sim_time to be updated
            rclpy.spin_once(self.node, timeout_sec=0.5)

            current_sim_time = copy.copy(self._current_sim_time)
            time_diff_ns = current_sim_time.nanoseconds - self._previous_update_sim_time.nanoseconds
            if time_diff_ns < 0:
                # print("Negative time difference detected, probably due to a reset")
                self._previous_update_sim_time = rclpyTime()
                time_diff_ns = current_sim_time.nanoseconds
            loop_end_time = time.time()
            loop_duration = loop_end_time - loop_start_time

            if time_diff_ns >= self._update_period_ns or loop_duration > timeout_duration:
                break

        if loop_duration >= timeout_duration:
            srv_time = self._get_current_sim_time_from_srv()
            self.node.get_logger().warn(f"Wait for simulation loop timeout, getting time from service instead")
            self.node.get_logger().warn(f'Current sim time: {current_sim_time.nanoseconds}, time from service: {srv_time.nanoseconds}')
            self._current_sim_time = srv_time
            for i in range(20):
                rclpy.spin_once(self.node, timeout_sec=0.1)
                current_sim_time = copy.copy(self._current_sim_time)
        self._previous_update_sim_time = current_sim_time

    def _get_current_sim_time_from_srv(self) -> rclpyTime:
        client = self.node.create_client(GetCurrentSimTime, "/get_current_sim_time")
        req = GetCurrentSimTime.Request()
        retry_count = 0
        while not client.wait_for_service(timeout_sec=1.0) and retry_count < 10:
            self.node.get_logger().info('/get_current_sim_time service not available, waiting again...')
            retry_count += 1

        future = client.call_async(req)
        rclpy.spin_until_future_complete(self.node, future)
        if future.result() is not None:
            current_sim_time_sec = future.result().sec
            current_sim_time_nsec = future.result().nanosec
            current_sim_time = rclpyTime(seconds=current_sim_time_sec, nanoseconds=current_sim_time_nsec)
            return current_sim_time
        else:
            self.node.get_logger().warn('/get_current_sim_time service call failed')
            return rclpyTime()
