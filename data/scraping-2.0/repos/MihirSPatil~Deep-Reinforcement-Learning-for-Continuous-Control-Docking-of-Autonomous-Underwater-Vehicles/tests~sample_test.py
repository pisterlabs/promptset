import pytest
import numpy as np
from openai_ros.robot_envs import deepleng_env

@pytest.fixture
def env():
    return deepleng_env.DeeplengEnv()

@pytest.fixture
def quaternion():
    return np.array([0.06146124, 0, 0, 0.99810947])

def test_quaternion2euler(env, quaternion):
    angles = env.quaternion2euler(quaternion)
    assert np.allclose(angles, [0.123, 0, 0])