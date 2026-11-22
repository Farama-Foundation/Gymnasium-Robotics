import itertools
import numpy as np
import pytest

from gymnasium_robotics.utils import rotations


def test_point_euler_round_trip_single():
    euler = np.array([0.1, -0.5, 1.2])
    pe = rotations.euler2point_euler(euler)
    assert pe.shape == (1, 6)
    recovered = rotations.point_euler2euler(pe)
    assert recovered.shape == (1, 3)
    np.testing.assert_allclose(recovered[0], euler, atol=1e-7)


def test_point_euler_round_trip_all_quadrants():
    test_angles = [
        -np.pi,
        -3 * np.pi / 4,
        -np.pi / 2,
        -np.pi / 4,
        0.0,
        np.pi / 4,
        np.pi / 2,
        3 * np.pi / 4,
        np.pi,
    ]
    # Build a batch of Euler angle triplets covering all combinations
    triplets = np.array(list(itertools.product(test_angles, repeat=3)))
    pe = rotations.euler2point_euler(triplets)
    assert pe.shape == (len(triplets), 6)
    recovered = rotations.point_euler2euler(pe)
    assert recovered.shape == (len(triplets), 3)

    # Convert back to sin/cos to compare angles mod 2*pi
    np.testing.assert_allclose(np.sin(recovered), np.sin(triplets), atol=1e-7)
    np.testing.assert_allclose(np.cos(recovered), np.cos(triplets), atol=1e-7)
    # And directly check values in [-pi, pi]
    np.testing.assert_allclose(rotations.normalize_angles(recovered), rotations.normalize_angles(triplets), atol=1e-7)


def test_point_euler_cosine_zero_divisions():
    # Angles with cos == 0, e.g. pi/2 and -pi/2
    pe = np.array([[1.0, -1.0, 0.0, 0.0, 0.0, -1.0]])
    recovered = rotations.point_euler2euler(pe)
    expected = np.array([[np.pi / 2, -np.pi / 2, np.pi]])
    np.testing.assert_allclose(recovered, expected, atol=1e-7)
