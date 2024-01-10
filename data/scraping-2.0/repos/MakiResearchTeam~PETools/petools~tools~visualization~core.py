import cv2
import numpy as np

from petools.tools import Human
from .coherence_check import coherence_check
from petools.tools.estimate_tools.constants import CONNECT_KP
from .config import TextDrawConfig


def _draw_human(
        image: np.ndarray,
        points: list,
        connect_indices: list,
        color: tuple = (255, 0, 0),
        thickness: int = 2,
        conf_threshold: float = 1e-3,
        pose_name: str = None,
        pose_name_position: tuple = (100, 100),
        pose_conf: float = None,
        pose_conf_position: tuple = (120, 120),
):
    """
    Draws a single human on the given image. Note that drawing happens inplace, so if you don't
    want the original image to be modified, pass a copy of it.

    Parameters
    ----------
    image : np.ndarray
        The image.
    points : array-like
        An array of shape [n_points, 3] containing (x, y) coordinates of the human's joints and their corresponding
        confidences (the 3rd dim).
    connect_indices : array-like
        Contains pairs of the points' indices showing how to connect them.
    color : tuple
        Color of the human to draw.
    thickness : int
        Thickness of the lines used to draw the human.
    conf_threshold : float
        Minimum threshold a pair of points must pass to be drawn.
    pose_name : str, optional
        Name of the pose class.
    pose_name_position : tuple
        Position where to draw the pose name.
    pose_conf : float, optional
        Confidence of the given pose (class) name.
    pose_conf_position : tuple
        Position where to draw the pose name.
    """
    for i in range(len(connect_indices)):
        ind1, ind2 = connect_indices[i]
        p1 = points[ind1]
        p2 = points[ind2]

        if p1[2] < conf_threshold or p2[2] < conf_threshold:
            continue

        line_p1 = int(p1[0]), int(p1[1])
        line_p2 = int(p2[0]), int(p2[1])
        cv2.line(image, line_p1, line_p2, color=color, thickness=thickness)

    if pose_name:
        cv2.putText(
            image,
            pose_name,
            pose_name_position,
            TextDrawConfig.POSE_NAME_FONT,
            TextDrawConfig.POSE_NAME_FONT_SCALE,
            TextDrawConfig.POSE_NAME_FONT_COLOR,
            TextDrawConfig.POSE_NAME_FONT_THICKNESS
        )

    if pose_name:
        cv2.putText(
            image,
            str(pose_conf),
            pose_conf_position,
            TextDrawConfig.POSE_CONF_FONT,
            TextDrawConfig.POSE_CONF_FONT_SCALE,
            TextDrawConfig.POSE_CONF_FONT_COLOR,
            TextDrawConfig.POSE_CONF_FONT_THICKNESS
        )


def draw_human(
        image: np.ndarray,
        human,
        connect_indices: list = CONNECT_KP,
        color: tuple = (255, 0, 0),
        thickness: int = 2,
        conf_threshold: float = 1e-3,
        pose_name: str = None,
        pose_name_position: tuple = (100, 100),
        pose_conf: float = None,
        pose_conf_position: tuple = (120, 120),
        do_coherence_check: bool = True,
        root_points: tuple = (2, 0, 1, 4, 5, 10, 11, 22)
):
    """
    Draws a single human on the given image. Note that drawing happens inplace, so if you don't
    want the original image to be modified, pass a copy of it.

    Parameters
    ----------
    image : np.ndarray
        The image.
    human : list, dict, Human
        The human to draw. Elements of the structure must be of length 3 (x, y, conf).
    connect_indices : array-like
        Contains pairs of the points' indices showing how to connect them.
    color : tuple
        Color of the human to draw.
    thickness : int
        Thickness of the lines used to draw the human.
    conf_threshold : float
        Minimum threshold a pair of points must pass to be drawn.
    pose_name : str, optional
        Name of the pose class.
    pose_name_position : tuple
        Position where to draw the pose name.
    pose_conf : float, optional
        Confidence of the given pose (class) name.
    pose_conf_position : tuple
        Position where to draw the pose name.
    do_coherence_check : bool
        Whether to perform human skeleton coherence check.
    root_points : tuple
        A list of root points used in the coherence check. From those points tracing will be performed
        to mask the absent (unconnected) part of the skeleton graph.
    """
    if isinstance(human, Human):
        points = human.to_np()
    elif isinstance(human, list):
        assert len(human[0]) == 3, \
            f'The list must contain elements of length 3, but received elem={human[0]}'
        points = human
    elif isinstance(human, dict):
        points = [val for val in human.values()]
        assert len(points[0]) == 3, \
            f'The dictionary must contain elements of length 3, but received len(elem)={points[0]}'
    elif isinstance(human, np.ndarray):
        points = human
    else:
        raise Exception("Unrecognized data type. Expected data types: np.ndarray, list, dict, Human. "
                        f"Received type: {type(human)}. \nData:\n {human}")

    if do_coherence_check:
        points = coherence_check(
            points=points,
            root_points=root_points,
            connect_indices=connect_indices,
            conf_threshold=conf_threshold
        )

    _draw_human(
        image=image,
        points=points,
        connect_indices=connect_indices,
        color=color,
        thickness=thickness,
        conf_threshold=conf_threshold,
        pose_name=pose_name,
        pose_name_position=pose_name_position,
        pose_conf=pose_conf,
        pose_conf_position=pose_conf_position
    )
