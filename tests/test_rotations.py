import mujoco
import numpy as np
import pytest

from gymnasium_robotics.utils import rotations


@pytest.mark.parametrize("batch_shape", [(), (5,), (2, 3), (4, 4), (0,), (2, 0)])
def test_quat_mul_preserves_batch_axes(batch_shape):
    """Quaternion components stay on the last axis for every batch shape."""
    rng = np.random.default_rng(0)
    q0 = rotations.euler2quat(rng.uniform(-np.pi, np.pi, batch_shape + (3,)))
    q1 = rotations.euler2quat(rng.uniform(-np.pi, np.pi, batch_shape + (3,)))
    expected = np.empty_like(q0)
    for index in np.ndindex(batch_shape):
        mujoco.mju_mulQuat(expected[index], q0[index], q1[index])

    result = rotations.quat_mul(q0, q1)

    assert result.shape == q0.shape
    np.testing.assert_allclose(result, expected, atol=1e-12)
