import pytest
import numpy as np
from openai_ros.robot_envs.transform_utils import transformed_nose_position

@pytest.fixture
def orientation():
    return np.array([0.0, 0.0, 0.0, 0.0])

@pytest.fixture
def position_center():
    return np.array([5, 0, 0])


@pytest.fixture
def position_nose():
    return np.array([3.65, 0, 0])


@pytest.fixture
def nose_in_body():
    return np.array([1.35, 0, 0])


def test_transformed_nose_position_setter(position_center,
                                          position_nose,
                                          orientation,
                                          nose_in_body):
    assert np.allclose(position_center - transformed_nose_position(orientation, nose_in_body), position_nose)


def test_transformed_nose_position_getter(position_center,
                                          position_nose,

                                          orientation,
                                          nose_in_body):
    assert np.allclose(position_nose + transformed_nose_position(orientation, nose_in_body), position_center)
