from enum import Enum
from typing import Dict

import numpy

from ros2_control_interfaces.msg import JointControl

from openai_ros2.robots import LobotArmSim

from gym.spaces import MultiDiscrete


class LobotArmSimDiscrete(LobotArmSim):
    class Action(Enum):
        BigPositive = 0, 0.05
        SmallPositive = 1, 0.01
        Remain = 2, 0
        SmallNegative = 3, -0.05
        BigNegative = 4, -0.01

        def __new__(cls, value, corresponding_value):
            member = object.__new__(cls)
            member._value_ = value
            member.corresponding_value = corresponding_value
            return member

        def __int__(self):
            return self.value

    '''-------------PUBLIC METHODS START-------------'''

    def __init__(self, node, robot_kwargs: Dict = None):
        if robot_kwargs is None:
            robot_kwargs = {}
        super().__init__(node, robot_kwargs)

    def set_action(self, action: numpy.ndarray) -> None:
        """
        Sets the action, unpauses the simulation and then waits until the update period of openai gym is over.
        The simulation is also expected to pause at the same time.
        This is to create a deterministic time step for the gym environment such that the agent can properly evaluate
        each action
        :param action:
        :return: obs, reward, done, info
        """
        assert len(action) == 3, f"{len(action)} actions passed to LobotArmSim, expected: 3"
        assert action.max() <= 4, "Max of action space more than 4"
        assert action.min() >= 0, "Min of action space less than 0"
        assert action.dtype == int

        action_values = [LobotArmSimDiscrete.Action(x).corresponding_value for x in action]
        self._target_joint_state += action_values

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
        self._reset_state()
        # No unpause here because it is assumed that the set_action will unpause it

    def get_action_space(self):
        possible_action_count = len(LobotArmSimDiscrete.Action)
        return MultiDiscrete([possible_action_count, possible_action_count, possible_action_count])

    '''-------------PUBLIC METHODS END-------------'''
