"""File for ManySegmentSwimmerEnv.

This file is originally from the `schroederdewitt/multiagent_mujoco` repository hosted on GitHub
(https://github.com/schroederdewitt/multiagent_mujoco/blob/master/multiagent_mujoco/manyagent_swimmer.py)
Original Author: Schroeder de Witt

 - General code cleanup, factorization, type hinting, adding documentation and comments
 - updated API to Gymnasium.MuJoCo v4
 - increase returned info
 - renamed ManyAgentSwimmerEnv -> ManySegmentSwimmerEnv (and changed the __init__ arguments accordingly)
"""


import os
from jinja2 import Template


def gen_asset(n_segs: int, asset_path: str) -> None:
    template_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "assets",
        "many_segment_swimmer.xml.template",
    )
    with open(template_path) as file:
        template = Template(file.read())
        body_str_template = """
        <body name="mid{:d}" pos="-1 0 0">
          <geom density="1000" fromto="0 0 0 -1 0 0" size="0.1" type="capsule"/>
          <joint axis="0 0 {:d}" limited="true" name="rot{:d}" pos="0 0 0" range="-100 100" type="hinge"/>
        """

        body_end_str_template = """
        <body name="back" pos="-1 0 0">
            <geom density="1000" fromto="0 0 0 -1 0 0" size="0.1" type="capsule"/>
            <joint axis="0 0 1" limited="true" name="rot{:d}" pos="0 0 0" range="-100 100" type="hinge"/>
          </body>
        """

        body_close_str_template = "</body>\n"
        actuator_str_template = """\t <motor ctrllimited="true" ctrlrange="-1 1" gear="150.0" joint="rot{:d}"/>\n"""

        body_str = ""
        for i in range(1, n_segs - 1):
            body_str += body_str_template.format(i, (-1) ** (i + 1), i)
        body_str += body_end_str_template.format(n_segs - 1)
        body_str += body_close_str_template * (n_segs - 2)

        actuator_str = ""
        for i in range(n_segs):
            actuator_str += actuator_str_template.format(i)

        rt = template.render(body=body_str, actuators=actuator_str)
        with open(asset_path, "w") as file:
            file.write(rt)
