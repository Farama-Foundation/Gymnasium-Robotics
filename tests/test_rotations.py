"""Regression tests for rotation utilities."""

import numpy as np

from gymnasium_robotics.utils import rotations


def test_quat_rot_vec_matches_matrix_action():
    q = rotations.euler2quat(np.arange(18).reshape(2, 3, 3) / 10)
    v = np.arange(18).reshape(2, 3, 3)
    expected = (rotations.quat2mat(q) @ v[..., None])[..., 0]
    np.testing.assert_allclose(rotations.quat_rot_vec(q, v), expected, atol=1e-12)
