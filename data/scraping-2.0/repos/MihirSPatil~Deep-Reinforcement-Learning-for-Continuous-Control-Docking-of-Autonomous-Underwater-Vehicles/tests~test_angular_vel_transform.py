import pytest
import numpy as np
from tf.transformations import euler_from_quaternion
from openai_ros.robot_envs.transform_utils import angular_transform, angular_to_body, angular_to_world

@pytest.fixture
def orientation():
    return np.array([0.0, 0.0, 0.26, 0.97])

@pytest.fixture
def world_ang_vel():
    return np.array([0.257969377375, 0.0840644014825, -0.234474043443])

@pytest.fixture
def body_ang_vel():
    return np.array([ 0.25796938, 0.0840644, -0.23447404])



def test_angular_to_world(world_ang_vel, body_ang_vel, orientation):
    roll, pitch, yaw = euler_from_quaternion(orientation)
    assert np.allclose(angular_transform(roll, pitch, yaw, frame="body2world").dot(body_ang_vel), world_ang_vel)


def test_angular_to_body(world_ang_vel, body_ang_vel, orientation):
    roll, pitch, yaw = euler_from_quaternion(orientation)
    assert np.allclose(angular_transform(roll, pitch, yaw, frame="world2body").dot(world_ang_vel), body_ang_vel)
