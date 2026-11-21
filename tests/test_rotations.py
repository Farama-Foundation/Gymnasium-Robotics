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


def test_quat2axisangle_half_turn_roundoff():
    q = np.array([0, np.nextafter(1.0, 2.0), 0, 0])
    with np.errstate(invalid="raise"):
        axis, angle = rotations.quat2axisangle(q)
    np.testing.assert_allclose(axis, [1, 0, 0])
    np.testing.assert_allclose(angle, np.pi)


def test_quat2point_quat_identity_in_batch():
    with np.errstate(divide="raise", invalid="raise"):
        points = rotations.quat2point_quat(np.array([[1., 0, 0, 0], [0., 0, 1, 0]]))
    np.testing.assert_allclose(points, [[0, 1, 0, 0, 0], [0, -1, 0, 1, 0]], atol=1e-12)


def test_mat2quat_empty_rollout():
    assert rotations.mat2quat(np.empty((2, 0, 3, 3))).shape == (2, 0, 4)


def test_unit_vector_list_into_buffer():
    out = np.empty(3)
    assert rotations.unit_vector([3., 4., 0.], out=out) is None
    np.testing.assert_allclose(out, [0.6, 0.8, 0.])
