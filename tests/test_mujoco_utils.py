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


def test_reset_mocap_welds_resets_relpose_without_changing_anchor():
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <body name="mocap" mocap="true"/>
            <body name="target">
              <freejoint/>
              <geom type="sphere" size="0.01"/>
            </body>
            <body name="other" pos="1 0 0">
              <freejoint/>
              <geom type="sphere" size="0.01"/>
            </body>
          </worldbody>
          <equality>
            <weld name="mocap_weld" body1="mocap" body2="target" anchor="0.1 0.2 0.3"/>
            <connect name="other_connect" body1="target" body2="other" anchor="0.2 0.3 0.4"/>
          </equality>
        </mujoco>
        """
    )
    data = mujoco.MjData(model)
    weld_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "mocap_weld")
    connect_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "other_connect")
    anchor = model.eq_data[weld_id, :3].copy()
    connect_data = model.eq_data[connect_id].copy()
    model.eq_data[weld_id, 3:10] = [1.0, 2.0, 3.0, 0.0, 0.4, 0.5, 0.6]

    mujoco_utils.reset_mocap_welds(model, data)

    np.testing.assert_array_equal(model.eq_data[weld_id, :3], anchor)
    np.testing.assert_array_equal(model.eq_data[weld_id, 3:6], anchor)
    np.testing.assert_array_equal(model.eq_data[weld_id, 6:10], [1.0, 0.0, 0.0, 0.0])
    np.testing.assert_array_equal(model.eq_data[connect_id], connect_data)
    # The first six constraint rows belong to the weld between the aligned bodies.
    np.testing.assert_allclose(data.efc_pos[:6], 0.0, atol=1e-12)
