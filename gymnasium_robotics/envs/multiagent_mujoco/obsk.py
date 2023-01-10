"""file Containing utily functions for MaMuJoCo.

This file is originally from the `schroederdewitt/multiagent_mujoco` repository hosted on GitHub
(https://github.com/schroederdewitt/multiagent_mujoco/blob/master/multiagent_mujoco/obsk.py)
Original Author: Schroeder de Witt

Then Modified by @Kallinteris-Andreas for this project
changes:
 - General code cleanup, factorization, type hinting, adding documentation and comments
 - `build_obs`: fixed global observations, fixed body observations (cvel, cinert, cfrc_ext), how uses mujoco.data, instead of gym.env
 - `HalfCheetah`: fix action ordering
 - `Ant`: Fix global observation, fix "2x4d" factorization how having diagonal observations
 - `Humanoid`s: Added Body support, fixed abdomen observations, added/fixed missing global torso observations, fixed action ordering
 - `Reacher`: Fixxed body mapping
 - `Pusher`: Added support for `Pusher`
 - `Swimmer`: Added Front tip to global observations
 - `Walker2D`: Added missing Global Observations
 - `CoupledHalfCheetah`: improved node naming, fixed tendon Jacobian observations, fixed action mapping of the second cheetah, added missing global observationsm, fixed action ordering
 - `ManySegmentAnt`: Fixed Global Observations
 - added new functions: `_observation_structure`

This project is covered by the Apache 2.0 License.
"""

from __future__ import annotations

import itertools
import typing
from copy import deepcopy

import numpy as np


class Node:
    """A node of the mujoco graph representing a single body part and it's corresponding single action & observetions."""

    def __init__(
        self,
        label: str,
        qpos_ids: int | None,
        qvel_ids: int | None,
        act_ids: int | None,
        body_fn: typing.Callable | None = None,
        bodies: tuple[int, ...] = (),
        extra_obs: dict[str, typing.Callable] = {},
        tendons: tuple[int, ...] = (),
    ):
        """Init.

        Args:
            label: the name of the node
            qpos_ids: the corresponding position ID,
            qvel_ids: the corresponding velocity ID,
            act_ids: the action's ID associated with that node
            body_fn: an optional overwrite of the bodies's functions
            bodies: is used to index ["cvel", "cinert", "cfrc_ext"] categories
            extra_obs: an optional overwrite of observation types keyied by categories
            tendons: the of list of tendon IDs
        """
        self.label = label
        self.qpos_ids = qpos_ids
        self.qvel_ids = qvel_ids
        self.act_ids = act_ids
        self.bodies = bodies
        self.extra_obs = extra_obs
        self.body_fn = body_fn
        self.tendons = tendons

    def __str__(self):
        """Returns a string of the node using the provided label."""
        return self.label

    def __repr__(self):
        """Returns a string of the node using the provided label."""
        return self.label


class HyperEdge:
    """A collection of nodes, that are fully connected (with edges).

    If a HyperEdge consists of 2 Nodes, then it is simply an Edge of those Nodes.

    More at: https://en.wikipedia.org/wiki/Hypergraph
    """

    def __init__(self, *nodes: Node):
        """Init.

        Args:
            nodes: the nodes that are fully connected
        """
        self.nodes = set(nodes)

    def __contains__(self, item: Node):
        """Checks if Item is in the edge."""
        return item in self.nodes

    def __str__(self):
        """Returns a string of the HyperEdge showing all the nodes in it."""
        return f"HyperEdge({self.nodes})"

    def __repr__(self):
        """Returns a string of the HyperEdge showing all the nodes in it."""
        return f"HyperEdge({self.nodes})"


def get_joints_at_kdist(
    agent_partition: tuple[Node, ...],
    hyperedges: list[HyperEdge],
    k: int,
) -> dict[int, list[Node]]:
    """Identify all joints at distance <= k from agent.

    Args:
        agent_partition:
            tuples of nodes of an agent
        hyperedges:
            hyperedges of the graph
        k:
            kth degree (number of nearest joints to observe)

    Returns:
        dict with k as key, and list of joints/nodes at that distance
    """

    def _adjacent(lst):  # return all sets adjacent to any element in lst
        ret = set()
        for element in lst:
            ret = ret.union(
                set(
                    itertools.chain(
                        *[
                            e.nodes.difference({element})
                            for e in hyperedges
                            if element in e
                        ]
                    )
                )
            )
        return ret

    explored_nodes = set(agent_partition)
    new_nodes = explored_nodes
    k_dict = {0: sorted(list(new_nodes), key=lambda x: x.label)}
    for key in range(1, k + 1):
        new_nodes = _adjacent(new_nodes) - explored_nodes
        explored_nodes = explored_nodes.union(new_nodes)
        k_dict[key] = sorted(list(new_nodes), key=lambda x: x.label)

    # assert that the nodes in `k_dict` are unique
    list_of_nodes = [item for sublist in list(k_dict.values()) for item in sublist]
    assert len(list_of_nodes) == len(set(list_of_nodes)), k_dict

    return k_dict


def build_obs(
    data,
    k_dict: dict[int, list[Node]],
    k_categories: list[list[str]],
    global_nodes: list[Node],
    global_categories: tuple[str, ...],
) -> np.ndarray:
    """Given a k_dict from get_joints_at_kdist, extract observation vector.

    Args:
        data: a structure containing the global state of the agent
        k_dict: the k_dict of an agent
        k_categories: the categories at every depth level
        global_nodes: The MuJoCo global godes
        global_categories: The observation Categories for the global MuJoCo nodes

    Returns:
        observation for the agent (indicated by K_dict)
    """
    body_set_dict = {}
    obs_lst = []
    # Add local observations
    for k in sorted(list(k_dict.keys())):
        for node in k_dict[k]:
            for category in k_categories[k]:
                if category in node.extra_obs:
                    items = node.extra_obs[category](data).tolist()
                    obs_lst.extend(items if isinstance(items, list) else [items])
                elif category in ["qvel"]:  # this is a "joint velocity" item
                    obs_lst.extend([data.qvel[node.qvel_ids]])
                elif category in ["qpos"]:  # this is a "joint position" item
                    obs_lst.extend([data.qpos[node.qpos_ids]])
                elif category in ["qfrc_actuator"]:  # this is a "actuator forces" item
                    obs_lst.extend([data.qfrc_actuator[node.qvel_ids]])
                elif category in ["cvel", "cinert", "cfrc_ext"]:
                    # this is a "body position" item
                    for body in node.bodies:
                        if category not in body_set_dict:
                            body_set_dict[category] = set()
                        if body not in body_set_dict[category]:
                            items = getattr(data, category)[body].tolist()
                            if node.body_fn is not None:
                                items = node.body_fn(body, items)
                            obs_lst.extend(
                                items if isinstance(items, list) else [items]
                            )
                            body_set_dict[category].add(body)

    # Add global observations
    body_set_dict = {}
    for category in global_categories:
        for joint in global_nodes:
            if category in joint.extra_obs:
                items = joint.extra_obs[category](data).tolist()
                obs_lst.extend(items if isinstance(items, list) else [items])
            elif category in ["qvel", "qpos"]:
                items = getattr(data, category)[getattr(joint, f"{category}_ids")]
                obs_lst.extend(items if isinstance(items, list) else [items])
            elif category in ["qfrc_actuator"]:  # this is a "actuator forces" item
                obs_lst.extend([data.qfrc_actuator[joint.qvel_ids]])
            else:
                for body in joint.bodies:
                    if category not in body_set_dict:
                        body_set_dict[category] = set()
                    if body not in body_set_dict[category]:
                        items = getattr(data, category)[body].tolist()
                        if joint.body_fn is not None:
                            items = joint.body_fn(body, items)
                        obs_lst.extend(items if isinstance(items, list) else [items])
                        body_set_dict[category].add(body)

    return np.array(obs_lst)


def get_parts_and_edges(  # noqa: C901
    label: str, partitioning: str | None
) -> tuple[list[tuple[Node, ...]], list[HyperEdge], list[Node]]:
    """Gets the mujoco Graph (nodes & edges) given an optional partitioning,.

    Args:
        label: the mujoco task to partition
        partitioning: the partioneing scheme

    Returns:
        the partition of the mujoco graph nodes, the graph edges, and global nodes
    """
    if label in ["HalfCheetah-v4"]:

        # define Mujoco graph
        bthigh = Node("bthigh", -6, -6, 0)
        bshin = Node("bshin", -5, -5, 1)
        bfoot = Node("bfoot", -4, -4, 2)
        fthigh = Node("fthigh", -3, -3, 3)
        fshin = Node("fshin", -2, -2, 4)
        ffoot = Node("ffoot", -1, -1, 5)

        edges = [
            HyperEdge(bfoot, bshin),
            HyperEdge(bshin, bthigh),
            HyperEdge(bthigh, fthigh),
            HyperEdge(fthigh, fshin),
            HyperEdge(fshin, ffoot),
        ]

        root_x = Node(
            "root_x", 0, 0, None, extra_obs={"qpos": lambda data: np.array([])}
        )
        root_z = Node("root_z", 1, 1, None)
        root_y = Node("root_y", 2, 2, None)
        globals = [root_x, root_z, root_y]

        if partitioning is None:
            parts = [(bthigh, bshin, bfoot, fthigh, fshin, ffoot)]
        elif partitioning == "2x3":
            parts = [(bthigh, bshin, bfoot), (fthigh, fshin, ffoot)]
        elif partitioning == "6x1":
            parts = [(bthigh,), (bshin,), (bfoot,), (fthigh,), (fshin,), (ffoot,)]
        else:
            raise Exception(f"UNKNOWN partitioning config: {partitioning}")

        return parts, edges, globals

    elif label in ["Ant-v4"]:

        # define Mujoco graph
        torso = 1
        front_left_leg = 2
        aux_1 = 3
        ankle_1 = 4
        front_right_leg = 5
        aux_2 = 6
        ankle_2 = 7
        back_leg = 8
        aux_3 = 9
        ankle_3 = 10
        right_back_leg = 11
        aux_4 = 12
        ankle_4 = 13

        hip1 = Node(  # front left leg
            "hip1",
            -8,
            -8,
            2,
            bodies=(torso, front_left_leg),
            body_fn=lambda _id, x: np.clip(x, -1, 1).tolist(),
        )
        ankle1 = Node(
            "ankle1",
            -7,
            -7,
            3,
            bodies=(front_left_leg, aux_1, ankle_1),
            body_fn=lambda _id, x: np.clip(x, -1, 1).tolist(),
        )
        hip2 = Node(  # front right leg
            "hip2",
            -6,
            -6,
            4,
            bodies=(torso, front_right_leg),
            body_fn=lambda _id, x: np.clip(x, -1, 1).tolist(),
        )
        ankle2 = Node(
            "ankle2",
            -5,
            -5,
            5,
            bodies=(front_right_leg, aux_2, ankle_2),
            body_fn=lambda _id, x: np.clip(x, -1, 1).tolist(),
        )
        hip3 = Node(  # back left leg
            "hip3",
            -4,
            -4,
            6,
            bodies=(torso, back_leg),
            body_fn=lambda _id, x: np.clip(x, -1, 1).tolist(),
        )
        ankle3 = Node(
            "ankle3",
            -3,
            -3,
            7,
            bodies=(back_leg, aux_3, ankle_3),
            body_fn=lambda _id, x: np.clip(x, -1, 1).tolist(),
        )
        hip4 = Node(  # back right leg
            "hip4",
            -2,
            -2,
            0,
            bodies=(torso, right_back_leg),
            body_fn=lambda _id, x: np.clip(x, -1, 1).tolist(),
        )
        ankle4 = Node(
            "ankle4",
            -1,
            -1,
            1,
            bodies=(right_back_leg, aux_4, ankle_4),
            body_fn=lambda _id, x: np.clip(x, -1, 1).tolist(),
        )

        edges = [
            HyperEdge(ankle4, hip4),
            HyperEdge(ankle1, hip1),
            HyperEdge(ankle2, hip2),
            HyperEdge(ankle3, hip3),
            HyperEdge(hip4, hip1, hip2, hip3),
        ]

        torso = Node(
            "torso",
            0,
            0,
            None,
            extra_obs={
                "qpos": lambda data: data.qpos[2:7],
                "qvel": lambda data: data.qvel[:6],
                "cfrc_ext": lambda data: np.clip(data.cfrc_ext[0:1], -1, 1),
            },
        )
        globals = [torso]

        if partitioning is None:
            parts = [(hip4, ankle4, hip1, ankle1, hip2, ankle2, hip3, ankle3)]
        elif partitioning == "2x4":  # neighboring legs together (front and back)
            parts = [(hip1, ankle1, hip2, ankle2), (hip3, ankle3, hip4, ankle4)]
        elif partitioning == "2x4d":  # diagonal legs together
            parts = [(hip1, ankle1, hip4, ankle4), (hip2, ankle2, hip3, ankle3)]
        elif partitioning == "4x2":
            parts = [(hip1, ankle1), (hip2, ankle2), (hip3, ankle3), (hip4, ankle4)]
        else:
            raise Exception(f"UNKNOWN partitioning config: {partitioning}")

        return parts, edges, globals

    elif label in ["Hopper-v4"]:

        # define Mujoco-Graph
        thigh_joint = Node(
            "thigh_joint",
            -3,
            -3,
            0,
            extra_obs={
                "qvel": lambda data: np.clip(np.array([data.qvel[-3]]), -10, 10)
            },
        )
        leg_joint = Node(
            "leg_joint",
            -2,
            -2,
            1,
            extra_obs={
                "qvel": lambda data: np.clip(np.array([data.qvel[-2]]), -10, 10)
            },
        )
        foot_joint = Node(
            "foot_joint",
            -1,
            -1,
            2,
            extra_obs={
                "qvel": lambda data: np.clip(np.array([data.qvel[-1]]), -10, 10)
            },
        )

        edges = [HyperEdge(foot_joint, leg_joint), HyperEdge(leg_joint, thigh_joint)]

        root_x = Node(
            "root_x",
            0,
            0,
            None,
            extra_obs={
                "qpos": lambda data: np.array([]),  # Disable observation
                "qvel": lambda data: np.clip(np.array([data.qvel[0]]), -10, 10),
            },
        )
        root_z = Node(
            "root_z",
            1,
            1,
            None,
            extra_obs={"qvel": lambda data: np.clip(np.array([data.qvel[1]]), -10, 10)},
        )
        root_y = Node(
            "root_y",
            2,
            2,
            None,
            extra_obs={"qvel": lambda data: np.clip(np.array([data.qvel[2]]), -10, 10)},
        )
        globals = [root_x, root_z, root_y]

        if partitioning is None:
            parts = [
                (
                    thigh_joint,
                    leg_joint,
                    foot_joint,
                )
            ]
        elif partitioning == "3x1":
            parts = [(thigh_joint,), (leg_joint,), (foot_joint,)]

        else:
            raise Exception(f"UNKNOWN partitioning config: {partitioning}")

        return parts, edges, globals

    elif label in ["Humanoid-v4", "HumanoidStandup-v4"]:
        # bodies
        # worldbody = 0
        torso = 1
        lwaist = 2
        pelvis = 3
        right_thigh = 4
        right_sin = 5
        right_foot = 6
        left_thigh = 7
        left_sin = 8
        left_foot = 9
        right_upper_arm = 10
        right_lower_arm = 11
        left_upper_arm = 12
        left_lower_arm = 13

        # define Mujoco-Graph
        abdomen_y = Node("abdomen_y", -17, -17, 0, bodies=(torso, lwaist, pelvis))
        abdomen_z = Node("abdomen_z", -16, -16, 1, bodies=(torso, lwaist, pelvis))
        abdomen_x = Node(
            "abdomen_x", -15, -15, 2, bodies=(pelvis, right_thigh, left_thigh)
        )
        right_hip_x = Node("right_hip_x", -14, -14, 3, bodies=(right_thigh, right_sin))
        right_hip_z = Node("right_hip_z", -13, -13, 4, bodies=(right_thigh, right_sin))
        right_hip_y = Node("right_hip_y", -12, -12, 5, bodies=(right_thigh, right_sin))
        right_knee = Node("right_knee", -11, -11, 6, bodies=(right_sin, right_foot))
        left_hip_x = Node("left_hip_x", -10, -10, 7, bodies=(left_thigh, left_sin))
        left_hip_z = Node("left_hip_z", -9, -9, 8, bodies=(left_thigh, left_sin))
        left_hip_y = Node("left_hip_y", -8, -8, 9, bodies=(left_thigh, left_sin))
        left_knee = Node("left_knee", -7, -7, 10, bodies=(left_sin, left_foot))
        right_shoulder1 = Node(
            "right_shoulder1",
            -6,
            -6,
            11,
            bodies=(torso, right_upper_arm, right_lower_arm),
        )
        right_shoulder2 = Node(
            "right_shoulder2",
            -5,
            -5,
            12,
            bodies=(torso, right_upper_arm, right_lower_arm),
        )
        right_elbow = Node("right_elbow", -4, -4, 13, bodies=(right_lower_arm,))
        left_shoulder1 = Node(
            "left_shoulder1", -3, -3, 14, bodies=(torso, left_upper_arm, left_lower_arm)
        )
        left_shoulder2 = Node(
            "left_shoulder2", -2, -2, 15, bodies=(torso, left_upper_arm, left_lower_arm)
        )
        left_elbow = Node("left_elbow", -1, -1, 16, bodies=(left_lower_arm,))

        edges = [
            HyperEdge(abdomen_x, abdomen_y, abdomen_z),
            HyperEdge(right_hip_x, right_hip_y, right_hip_z),
            HyperEdge(left_hip_x, left_hip_y, left_hip_z),
            HyperEdge(left_elbow, left_shoulder1, left_shoulder2),
            HyperEdge(right_elbow, right_shoulder1, right_shoulder2),
            HyperEdge(left_knee, left_hip_x, left_hip_y, left_hip_z),
            HyperEdge(right_knee, right_hip_x, right_hip_y, right_hip_z),
            HyperEdge(left_shoulder1, left_shoulder2, abdomen_x, abdomen_y, abdomen_z),
            HyperEdge(
                right_shoulder1, right_shoulder2, abdomen_x, abdomen_y, abdomen_z
            ),
            HyperEdge(
                abdomen_x, abdomen_y, abdomen_z, left_hip_x, left_hip_y, left_hip_z
            ),
            HyperEdge(
                abdomen_x, abdomen_y, abdomen_z, right_hip_x, right_hip_y, right_hip_z
            ),
        ]

        root = Node(  # this is the torso
            "root",
            None,
            None,
            None,
            extra_obs={
                "qpos": lambda data: data.qpos[2:7],
                "qvel": lambda data: data.qvel[:6],
                "qfrc_actuator": lambda data: data.qfrc_actuator[:6],
                # "cfrc_ext": lambda data: np.clip(data.cfrc_ext[0:1], -1, 1),
            },
        )
        globals = [root]

        if partitioning is None:
            parts = [
                (
                    abdomen_x,
                    abdomen_y,
                    abdomen_z,
                    right_hip_x,
                    right_hip_y,
                    right_hip_z,
                    right_knee,
                    left_hip_x,
                    left_hip_y,
                    left_hip_z,
                    left_knee,
                    right_shoulder1,
                    right_shoulder2,
                    right_elbow,
                    left_shoulder1,
                    left_shoulder2,
                    left_elbow,
                ),
            ]
        elif partitioning == "9|8":  # isolate upper and lower body
            parts = [
                (  # Upper Body
                    abdomen_x,
                    abdomen_y,
                    abdomen_z,
                    right_shoulder1,
                    right_shoulder2,
                    right_elbow,
                    left_shoulder1,
                    left_shoulder2,
                    left_elbow,
                ),
                (  # Lower Body
                    right_hip_x,
                    right_hip_y,
                    right_hip_z,
                    right_knee,
                    left_hip_x,
                    left_hip_y,
                    left_hip_z,
                    left_knee,
                ),
            ]
            # TODO: There could be tons of decompositions here

        else:
            raise Exception(f"UNKNOWN partitioning config: {partitioning}")

        return parts, edges, globals

    elif label in ["Reacher-v4"]:
        # define Mujoco-Graph
        # worldbody = 0
        body0 = 1
        body1 = 2
        fingertip = 3
        # target = 4
        joint0 = Node(
            "joint0",
            -4,
            -4,
            0,
            bodies=(body0, body1),
            extra_obs={
                "qpos": (
                    lambda data: np.array(
                        [
                            np.sin(data.qpos[-4]),
                            np.cos(data.qpos[-4]),
                        ]
                    )
                )
            },
        )
        joint1 = Node(
            "joint1",
            -3,
            -3,
            1,
            bodies=(body1, fingertip),
            extra_obs={
                "fingertip_dist": (
                    lambda data: data.body("fingertip").xpos - data.body("target").xpos
                ),
                "qpos": (
                    lambda data: np.array(
                        [
                            np.sin(data.qpos[-3]),
                            np.cos(data.qpos[-3]),
                        ]
                    )
                ),
            },
        )
        edges = [HyperEdge(joint0, joint1)]

        target_x = Node(
            "target_x", -2, -2, None, extra_obs={"qvel": (lambda data: np.array([]))}
        )
        target_y = Node(
            "target_y", -1, -1, None, extra_obs={"qvel": (lambda data: np.array([]))}
        )
        globals = [target_x, target_y]

        if partitioning is None:
            parts = [
                (
                    joint0,
                    joint1,
                )
            ]
        elif partitioning == "2x1":
            # isolate upper and lower arms
            parts = [(joint0,), (joint1,)]
            # TODO: There could be tons of decompositions here

        else:
            raise Exception(f"UNKNOWN partitioning config: {partitioning}")

        return parts, edges, globals

    elif label in ["Pusher-v4"]:
        # define Mujoco-Graph
        r_shoulder_pan_joint = Node("r_wrist_roll_joint", 0, 0, 0)
        r_shoulder_lift_joint = Node("r_wrist_roll_joint", 1, 1, 1)
        r_upper_arm_roll_joint = Node("r_upper_arm_roll_joint", 2, 2, 2)
        r_elbow_flex_joint = Node("r_elbow_flex_joint", 3, 3, 3)
        r_forearm_roll_joint = Node("r_forearm_roll_joint", 4, 4, 4)
        r_wrist_flex_joint = Node("r_wrist_flex_joint", 5, 5, 5)
        r_wrist_roll_joint = Node("r_wrist_roll_joint", 6, 6, 6)

        edges = [
            HyperEdge(r_shoulder_pan_joint, r_shoulder_lift_joint),
            HyperEdge(r_shoulder_lift_joint, r_upper_arm_roll_joint),
            HyperEdge(r_upper_arm_roll_joint, r_elbow_flex_joint),
            HyperEdge(r_elbow_flex_joint, r_forearm_roll_joint),
            HyperEdge(r_forearm_roll_joint, r_wrist_flex_joint),
            HyperEdge(r_wrist_flex_joint, r_wrist_roll_joint),
        ]

        tips_arm_com = Node(
            "tips_arm",
            None,
            None,
            None,
            extra_obs={
                "qpos": (lambda data: np.array(data.body("tips_arm").xpos)),
                "qvel": (lambda data: np.array([])),
            },
        )
        object_com = Node(
            "object",
            None,
            None,
            None,
            extra_obs={
                "qpos": (lambda data: np.array(data.body("object").xpos)),
                "qvel": (lambda data: np.array([])),
            },
        )
        goal_com = Node(
            "goal",
            None,
            None,
            None,
            extra_obs={
                "qpos": (lambda data: np.array(data.body("goal").xpos)),
                "qvel": (lambda data: np.array([])),
            },
        )

        globals = [tips_arm_com, object_com, goal_com]

        if partitioning is None:
            parts = [
                (
                    r_shoulder_pan_joint,
                    r_shoulder_lift_joint,
                    r_upper_arm_roll_joint,
                    r_elbow_flex_joint,
                    r_forearm_roll_joint,
                    r_wrist_flex_joint,
                    r_wrist_roll_joint,
                )
            ]
        if partitioning == "3p":
            parts = [
                (
                    r_shoulder_pan_joint,
                    r_shoulder_lift_joint,
                    r_upper_arm_roll_joint,
                ),  # Shoulder
                (r_elbow_flex_joint,),  # Elbow
                (r_forearm_roll_joint, r_wrist_flex_joint, r_wrist_roll_joint),  # Wrist
            ]
            # TODO: There could be tons of decompositions here
        else:
            raise Exception(f"UNKNOWN partitioning config: {partitioning}")

        return parts, edges, globals

    elif label in ["Swimmer-v4"]:
        # define Mujoco-Graph
        joint0 = Node(
            "rot2",
            -2,
            -2,
            0,
            extra_obs={"qvel": (lambda data: np.array([data.qvel[0], data.qvel[3]]))},
        )
        joint1 = Node(
            "rot3",
            -1,
            -1,
            1,
            extra_obs={"qvel": (lambda data: np.array([data.qvel[1], data.qvel[4]]))},
        )

        edges = [HyperEdge(joint0, joint1)]
        free_body_rot = Node("free_body_rot", 2, 2, None)
        globals = [free_body_rot]

        if partitioning is None:
            parts = [
                (
                    joint0,
                    joint1,
                )
            ]
        elif partitioning == "2x1":
            parts = [(joint0,), (joint1,)]
        else:
            raise Exception(f"UNKNOWN partitioning config: {partitioning}")

        return parts, edges, globals

    elif label in ["Walker2d-v4"]:
        # define Mujoco-Graph
        thigh_joint = Node("thigh_joint", -6, -6, 0)
        leg_joint = Node("leg_joint", -5, -5, 1)
        foot_joint = Node("foot_joint", -4, -4, 2)
        thigh_left_joint = Node("thigh_left_joint", -3, -3, 3)
        leg_left_joint = Node("leg_left_joint", -2, -2, 4)
        foot_left_joint = Node("foot_left_joint", -1, -1, 5)

        edges = [
            HyperEdge(foot_joint, leg_joint),
            HyperEdge(leg_joint, thigh_joint),
            HyperEdge(foot_left_joint, leg_left_joint),
            HyperEdge(leg_left_joint, thigh_left_joint),
            HyperEdge(thigh_joint, thigh_left_joint),
        ]
        root_x = Node(
            "root_x", 0, 0, None, extra_obs={"qpos": lambda data: np.array([])}
        )
        root_z = Node("root_z", 1, 1, None)
        root_y = Node("root_y", 2, 2, None)
        globals = [root_x, root_x, root_z]

        if partitioning is None:
            parts = [
                (
                    foot_joint,
                    leg_joint,
                    thigh_joint,
                    foot_left_joint,
                    leg_left_joint,
                    thigh_left_joint,
                ),
            ]
        elif partitioning == "2x3":  # isolate right and left foot
            parts = [
                (foot_joint, leg_joint, thigh_joint),
                (
                    foot_left_joint,
                    leg_left_joint,
                    thigh_left_joint,
                ),
            ]
            # TODO: There could be tons of decompositions here

        else:
            raise Exception(f"UNKNOWN partitioning config: {partitioning}")

        return parts, edges, globals

    elif label in ["CoupledHalfCheetah-v4"]:
        # define Mujoco graph
        tendon = 0

        bthigh0 = Node(
            "bthigh0",
            -6,
            -6,
            0,
            # tendons=(tendon),
            extra_obs={
                "ten_J": lambda data: np.concatenate(
                    [data.ten_J[tendon][:2], data.ten_J[tendon][9:11]]
                ),
                "ten_length": lambda data: data.ten_length[tendon],
                "ten_velocity": lambda data: data.ten_velocity[tendon],
            },
        )
        bshin0 = Node("bshin0", -5, -5, 1)
        bfoot0 = Node("bfoot0", -4, -4, 2)
        fthigh0 = Node("fthigh0", -3, -3, 3)
        fshin0 = Node("fshin0", -2, -2, 4)
        ffoot0 = Node("ffoot0", -1, -1, 5)

        bthigh1 = Node(
            "bthigh1",
            -6,
            -6,
            6,
            # tendons=(tendon),
            extra_obs={
                "ten_J": lambda data: np.concatenate(
                    [data.ten_J[tendon][:2], data.ten_J[tendon][9:11]]
                ),
                "ten_length": lambda data: data.ten_length[tendon],
                "ten_velocity": lambda data: data.ten_velocity[tendon],
            },
        )
        bshin1 = Node("bshin1", -5, -5, 7)
        bfoot1 = Node("bfoot1", -4, -4, 8)
        fthigh1 = Node("fthigh1", -3, -3, 9)
        fshin1 = Node("fshin1", -2, -2, 10)
        ffoot1 = Node("ffoot1", -1, -1, 11)

        edges = [
            HyperEdge(bfoot0, bshin0),
            HyperEdge(bshin0, bthigh0),
            HyperEdge(bthigh0, fthigh0),
            HyperEdge(fthigh0, fshin0),
            HyperEdge(fshin0, ffoot0),
            HyperEdge(bfoot1, bshin1),
            HyperEdge(bshin1, bthigh1),
            HyperEdge(bthigh1, fthigh1),
            HyperEdge(fthigh1, fshin1),
            HyperEdge(fshin1, ffoot1),
        ]

        root_x0 = Node(
            "root_x0", 0, 0, None, extra_obs={"qpos": lambda data: np.array([])}
        )
        root_z0 = Node("root_z0", 1, 1, None)
        root_y0 = Node("root_y0", 2, 2, None)
        root_x1 = Node(
            "root_x1", 9, 9, None, extra_obs={"qpos": lambda data: np.array([])}
        )
        root_z1 = Node("root_z1", 10, 10, None)
        root_y1 = Node("root_y1", 11, 11, None)
        globals = [root_x0, root_y0, root_z0, root_x1, root_y1, root_z1]

        if partitioning is None:
            parts = [
                (
                    bthigh0,
                    bshin0,
                    bfoot0,
                    fthigh0,
                    fshin0,
                    ffoot0,
                    bthigh1,
                    bshin1,
                    bfoot1,
                    fthigh1,
                    fshin1,
                    ffoot1,
                ),
            ]
        elif partitioning == "1p1":  # isolate the cheetahs
            parts = [
                (bthigh0, bshin0, bfoot0, fthigh0, fshin0, ffoot0),
                (bthigh1, bshin1, bfoot1, fthigh1, fshin1, ffoot1),
            ]
        else:
            raise Exception(f"UNKNOWN partitioning config: {partitioning}")

        return parts, edges, globals

    elif label in ["ManySegmentSwimmer-v4"]:
        assert partitioning is not None, "Partitioning, required with " + label

        try:
            n_agents = int(partitioning.split("x")[0])
            n_segs_per_agents = int(partitioning.split("x")[1])
            n_segs = n_agents * n_segs_per_agents
        except Exception:
            raise Exception(f"UNKNOWN partitioning config: {partitioning}")

        # Note: Default Swimmer corresponds to n_segs = 3

        # define Mujoco-Graph
        joints = [
            Node(f"rot{i:d}", -n_segs + i, -n_segs + i, i) for i in range(0, n_segs)
        ]
        edges = [HyperEdge(joints[i], joints[i + 1]) for i in range(n_segs - 1)]
        globals = []

        parts = [
            tuple(joints[i * n_segs_per_agents : (i + 1) * n_segs_per_agents])
            for i in range(n_agents)
        ]
        return parts, edges, globals

    elif label in ["ManySegmentAnt-v4"]:
        assert partitioning is not None, "Partitioning, required with " + label

        try:
            n_agents = int(partitioning.split("x")[0])
            n_segs_per_agents = int(partitioning.split("x")[1])
            n_segs = n_agents * n_segs_per_agents
        except Exception:
            raise Exception(f"UNKNOWN partitioning config: {partitioning}")

        edges: list[HyperEdge] = []
        joints = []
        hip1m = Node("Dummy_Node", None, None, None)
        hip2m = Node("Dummy_Node", None, None, None)
        for segment in range(n_segs):
            torso = 1 + segment * 7
            front_right_leg = 2 + segment * 7
            aux1 = 3 + segment * 7
            ankle1 = 4 + segment * 7
            back_leg = 5 + segment * 7
            aux2 = 6 + segment * 7
            ankle2 = 7 + segment * 7

            off = -4 * (n_segs - 1 - segment)
            hip1n = Node(
                f"hip1_{segment:d}",
                -4 - off,
                -4 - off,
                2 + 4 * segment,
                bodies=(torso, front_right_leg),
                body_fn=lambda _id, x: np.clip(x, -1, 1).tolist(),
            )
            ankle1n = Node(
                f"ankle1_{segment:d}",
                -3 - off,
                -3 - off,
                3 + 4 * segment,
                bodies=(front_right_leg, aux1, ankle1),
                body_fn=lambda _id, x: np.clip(x, -1, 1).tolist(),
            )
            hip2n = Node(
                f"hip2_{segment:d}",
                -2 - off,
                -2 - off,
                0 + 4 * segment,
                bodies=(torso, back_leg),
                body_fn=lambda _id, x: np.clip(x, -1, 1).tolist(),
            )
            ankle2n = Node(
                f"ankle2_{segment:d}",
                -1 - off,
                -1 - off,
                1 + 4 * segment,
                bodies=(back_leg, aux2, ankle2),
                body_fn=lambda _id, x: np.clip(x, -1, 1).tolist(),
            )

            edges += [
                HyperEdge(ankle1n, hip1n),
                HyperEdge(ankle2n, hip2n),
                HyperEdge(hip1n, hip2n),
            ]
            if segment:
                edges += [HyperEdge(hip1m, hip2m, hip1n, hip2n)]

            hip1m = deepcopy(hip1n)
            hip2m = deepcopy(hip2n)
            joints.append([hip1n, ankle1n, hip2n, ankle2n])

        free_joint = Node(
            "free",
            0,
            0,
            None,
            extra_obs={
                "qpos": lambda data: data.qpos[2:7],
                "qvel": lambda data: data.qvel[:6],
                "cfrc_ext": lambda data: np.clip(data.cfrc_ext[0:1], -1, 1),
            },
        )
        globals = [free_joint]

        parts = [
            [
                x
                for sublist in joints[
                    i * n_segs_per_agents : (i + 1) * n_segs_per_agents
                ]
                for x in sublist
            ]
            for i in range(n_agents)
        ]
        parts = [tuple(part) for part in parts]

        return parts, edges, globals
    else:
        raise Exception(f"UNKNOWN label environment: {label}")


def _observation_structure(scenario: str) -> dict[str, int]:
    """Get the types of observations for each Gymnasium.MuJoCo environment.

    Args:
        scenario: the mujoco scenartio

    Returns:
        a dictionary keyied by observation type with values indicating the number of observations for that type
    """
    ret = {
        "skipped_qpos": 0,  # Position data what is excluded/skip
        "qpos": 0,  # Position
        "qvel": 0,  # Velocity
        "cinert": 0,  # com inertia
        "cvel": 0,  # com velocity
        "qfrc_actuator": 0,  # Actuator Forces
        "cfrc_ext": 0,  # Contact Forces
    }

    if scenario == "Ant-v4":
        ret["skipped_qpos"] = 2
        ret["qpos"] = 13
        ret["qvel"] = 14
        # ret["cfrc_ext"] = 84
    elif scenario == "HalfCheetah-v4":
        ret["skipped_qpos"] = 1
        ret["qpos"] = 8
        ret["qvel"] = 9
    elif scenario == "Hopper-v4":
        ret["skipped_qpos"] = 1
        ret["qpos"] = 5
        ret["qvel"] = 6
    elif scenario == "HumanoidStandup-v4" or scenario == "Humanoid-v4":
        ret["skipped_qpos"] = 2
        ret["qpos"] = 22
        ret["qvel"] = 23
        ret["cinert"] = 140
        ret["cvel"] = 84
        ret["qfrc_actuator"] = 23
        ret["cfrc_ext"] = 84
    elif scenario == "InvertedDoublePendulum-v4":
        assert False, scenario + "can not be factorized"
        ret["qpos"] = 3
        ret["qvel"] = 3
        # qfrc_constraint = 3
    elif scenario == "InvertedPendulum-v4":
        assert False, scenario + "can not be factorized"
        ret["qpos"] = 2
        ret["qvel"] = 2
    elif scenario == "Pusher-v4":
        assert False, scenario + "is not supported"
        ret["qpos"] = 7
        ret["qvel"] = 7
        # 9 body_com
    elif scenario == "Reacher-v4":
        assert False, scenario + "can not be factorized"
        ret["qpos"] = 6
        ret["qvel"] = 2
        # 3 body_com
    elif scenario == "Swimmer-v4":
        ret["skipped_qpos"] = 2
        ret["qpos"] = 3
        ret["qvel"] = 5
    elif scenario == "Walker2d-v4":
        ret["skipped_qpos"] = 1
        ret["qpos"] = 8
        ret["qvel"] = 9

    return ret
