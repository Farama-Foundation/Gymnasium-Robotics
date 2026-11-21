"""Regression tests for rotation utilities."""

import numpy as np

from gymnasium_robotics.utils import rotations


def test_quat_rot_vec_matches_matrix_action():
    q = rotations.euler2quat(np.arange(18).reshape(2, 3, 3) / 10)
    v = np.arange(18).reshape(2, 3, 3)
    expected = (rotations.quat2mat(q) @ v[..., None])[..., 0]
    np.testing.assert_allclose(rotations.quat_rot_vec(q, v), expected, atol=1e-12)


def test_quat2mat_mixed_zero_and_nonzero_norms():
    with np.errstate(divide="raise", invalid="raise"):
        matrices = rotations.quat2mat([[0, 0, 0, 0], [0, 0, 0, 2]])
    np.testing.assert_array_equal(matrices, [np.eye(3), np.diag([-1, -1, 1])])
