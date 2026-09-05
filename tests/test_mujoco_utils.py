import mujoco
import numpy as np
import pytest

from gymnasium_robotics.utils import mujoco_utils


@pytest.mark.parametrize("joint_type", ["hinge", "slide"])
@pytest.mark.parametrize("field", ["qpos", "qvel"])
def test_scalar_joint_state(joint_type, field):
    """Read and write scalar joints without changing neighboring joint state."""
    model = mujoco.MjModel.from_xml_string(
        f"""
        <mujoco>
          <worldbody>
            <body>
              <joint name="neighbor" type="hinge"/>
              <geom size="0.1"/>
            </body>
            <body pos="1 0 0">
              <joint name="target" type="{joint_type}"/>
              <geom size="0.1"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    data = mujoco.MjData(model)
    state = getattr(data, field)
    state[:] = [0.25, 0.5]

    getter = getattr(mujoco_utils, f"get_joint_{field}")
    setter = getattr(mujoco_utils, f"set_joint_{field}")
    np.testing.assert_array_equal(getter(model, data, "target"), [0.5])
    setter(model, data, "target", 0.75)
    np.testing.assert_array_equal(state, [0.25, 0.75])
    np.testing.assert_array_equal(getter(model, data, "target"), [0.75])
