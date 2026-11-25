import itertools

import numpy as np
import pytest

from gymnasium_robotics.utils import rotations

AXES = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
    ]
)
AXES /= np.linalg.norm(AXES, axis=1, keepdims=True)


def axis_angle_quaternion(axis, angle):
    """Build a scalar-first unit quaternion from an axis and angle."""
    return np.concatenate(([np.cos(angle / 2)], axis * np.sin(angle / 2)))


@pytest.mark.parametrize("axis,angle", itertools.product(AXES, [0, np.pi / 2, np.pi]))
def test_point_quat_round_trip_single_rotation(axis, angle):
    """Preserve a single rotation and the API's leading batch dimension."""
    quaternion = axis_angle_quaternion(axis, angle)

    result = rotations.point_quat2quat(rotations.quat2point_quat(quaternion))

    assert result.shape == (1, 4)
    np.testing.assert_allclose(abs(result[0] @ quaternion), 1.0, atol=1e-7)


def test_point_quat_round_trip_batch_across_quadrants():
    """Preserve batched rotations across quadrants and equivalent signs."""
    boundary = np.finfo(float).eps
    angles = np.array(
        [
            -np.pi,
            -np.pi / 2 - boundary,
            -np.pi / 2,
            -np.pi / 2 + boundary,
            0,
            np.pi / 2 - boundary,
            np.pi / 2,
            np.pi / 2 + boundary,
            np.pi,
        ]
    )
    quaternions = np.array(
        [
            axis_angle_quaternion(axis, angle)
            for axis, angle in itertools.product(AXES, angles)
        ]
    )
    quaternions = np.concatenate((quaternions, -quaternions))

    result = rotations.point_quat2quat(rotations.quat2point_quat(quaternions))

    assert result.shape == quaternions.shape
    np.testing.assert_allclose(
        np.abs(np.sum(result * quaternions, axis=1)),
        np.ones(len(quaternions)),
        atol=1e-7,
    )
