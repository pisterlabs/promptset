import pytest
import numpy as np
from scipy.spatial.transform import Rotation as R
from openai_ros.robot_envs.transform_utils import rotation_from_quat


@pytest.fixture
def quaternion():
    quaternion = np.array([0.06146124,
                           0,
                           0,
                           0.99810947])
    return quaternion


def test_quaternion2euler(quaternion):
    rot_mat = rotation_from_quat(quaternion[0],
                                 quaternion[1],
                                 quaternion[2],
                                 quaternion[3])
    scipy_mat = R.from_quat([quaternion[0],
                            quaternion[1],
                            quaternion[2],
                            quaternion[3]]).as_matrix()

    assert np.allclose(rot_mat, scipy_mat)
