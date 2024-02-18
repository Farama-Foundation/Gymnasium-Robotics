"""File for ManySegmentAntEnv.

This file is originally from the `schroederdewitt/multiagent_mujoco` repository hosted on GitHub
(https://github.com/schroederdewitt/multiagent_mujoco/blob/master/multiagent_mujoco/manyagent_ant.py)
Original Author: Schroeder de Witt

 - General code cleanup, factorization, type hinting, adding documentation and comments
 - Removed the class (but kept the `gen_asset` function)
"""

import os

import gymnasium


def gen_asset(n_segs: int, asset_path: str) -> None:
    """Generates a variation of the Ant environment, but with ants coupled together (each segment has a torso + 4 legs).

    This environment was first introduced ["FACMAC: Factored Multi-Agent Centralised Policy Gradients"](https://arxiv.org/abs/2003.06709).
    """
    try:
        import jinja2
    except ImportError as e:
        raise gymnasium.error.dependencynotinstalled(
            f"{e}. "
            "(hint: you need to install jinja, run `pip install gymnasium_robotics[mamujoco]`.)"
        )

    template_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "assets",
        "many_segment_ant.xml.template",
    )

    with open(template_path) as file:
        template = jinja2.Template(file.read())

    body_str_template = """
    <body name="torso_{:d}" pos="-1 0 0">
       <!--<joint axis="0 1 0" name="nnn_{:d}" pos="0.0 0.0 0.0" range="-1 1" type="hinge"/>-->
        <geom density="100" fromto="1 0 0 0 0 0" size="0.1" type="capsule"/>
        <body name="front_right_leg_{:d}" pos="0 0 0">
          <geom fromto="0.0 0.0 0.0 0.0 0.2 0.0" name="aux1_geom_{:d}" size="0.08" type="capsule"/>
          <body name="aux_2_{:d}" pos="0.0 0.2 0">
            <joint axis="0 0 1" name="hip1_{:d}" pos="0.0 0.0 0.0" range="-30 30" type="hinge"/>
            <geom fromto="0.0 0.0 0.0 -0.2 0.2 0.0" name="right_leg_geom_{:d}" size="0.08" type="capsule"/>
            <body pos="-0.2 0.2 0">
              <joint axis="1 1 0" name="ankle1_{:d}" pos="0.0 0.0 0.0" range="-70 -30" type="hinge"/>
              <geom fromto="0.0 0.0 0.0 -0.4 0.4 0.0" name="right_ankle_geom_{:d}" size="0.08" type="capsule"/>
            </body>
          </body>
        </body>
        <body name="back_leg_{:d}" pos="0 0 0">
          <geom fromto="0.0 0.0 0.0 0.0 -0.2 0.0" name="aux2_geom_{:d}" size="0.08" type="capsule"/>
          <body name="aux2_{:d}" pos="0.0 -0.2 0">
            <joint axis="0 0 1" name="hip2_{:d}" pos="0.0 0.0 0.0" range="-30 30" type="hinge"/>
            <geom fromto="0.0 0.0 0.0 -0.2 -0.2 0.0" name="back_leg_geom_{:d}" size="0.08" type="capsule"/>
            <body pos="-0.2 -0.2 0">
              <joint axis="-1 1 0" name="ankle2_{:d}" pos="0.0 0.0 0.0" range="-70 -30" type="hinge"/>
              <geom fromto="0.0 0.0 0.0 -0.4 -0.4 0.0" name="third_ankle_geom_{:d}" size="0.08" type="capsule"/>
            </body>
          </body>
        </body>
    """

    body_close_str_template = "</body>\n"
    actuator_str_template = """\t     <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="hip1_{:d}" gear="150"/>
                                      <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="ankle1_{:d}" gear="150"/>
                                      <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="hip2_{:d}" gear="150"/>
                                      <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="ankle2_{:d}" gear="150"/>\n"""

    body_str = ""
    for i in range(1, n_segs):
        body_str += body_str_template.format(*([i] * 16))
    body_str += body_close_str_template * (n_segs - 1)

    actuator_str = ""
    for i in range(n_segs):
        actuator_str += actuator_str_template.format(*([i] * 8))

    rt = template.render(body=body_str, actuators=actuator_str)
    with open(asset_path, "w") as file:
        file.write(rt)
