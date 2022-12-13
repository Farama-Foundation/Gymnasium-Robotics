import itertools
import typing
from copy import deepcopy

import numpy
import numpy as np


class Node:
    def __init__(
        self,
        label,
        qpos_ids: int,
        qvel_ids: int,
        act_ids: int,
        body_fn=None,
        bodies: list[int] = None,
        extra_obs: dict[str : typing.Callable] = None,
        tendons=None,
    ):
        """
        A node of the mujoco graph representing a single body part and it's corresponding single action & observetions
        :param act_ids: the action assicaiated with that node
        :param extra_obs: an optional overwrite of observation types keyied by categories
        :param bodies: is used to index ["cvel", "cinert", "cfrc_ext"] categories
        """
        self.label = label
        self.qpos_ids = qpos_ids
        self.qvel_ids = qvel_ids
        self.act_ids = act_ids
        self.bodies = bodies
        self.extra_obs = {} if extra_obs is None else extra_obs
        self.body_fn = body_fn
        self.tendons = tendons
        pass

    def __str__(self):
        return self.label

    def __repr__(self):
        return self.label


class HyperEdge:
    def __init__(self, *edges: Node):
        self.edges = set(edges)

    def __contains__(self, item):
        return item in self.edges

    def __str__(self):
        return "HyperEdge({})".format(self.edges)

    def __repr__(self):
        return "HyperEdge({})".format(self.edges)


def get_joints_at_kdist(
    agent_partition: list[tuple[Node, ...]],
    hyperedges: list[HyperEdge],
    k: int = 0,
) -> dict[int : list[Node]]:
    """Identify all joints at distance <= k from agent agent_id

    :param agent_partition: tuples of nodes of an agent
    :param hyperedges: hyperedges of the graph
    :param k: kth degree (number of nearest joints to observe)
    :return:
        dict with k as key, and list of joints/nodes at that distance
    """

    if k is None:
        return None

    def _adjacent(lst):
        # return all sets adjacent to any element in lst
        ret = set()
        for element in lst:
            ret = ret.union(
                set(
                    itertools.chain(
                        *[
                            e.edges.difference({element})
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
    k_dict: dict[int : list[Node]],
    k_categories: list[list[str]],
    global_dict: dict[int : list[Node]],
    global_categories: list[list[str]],
) -> numpy.ndarray:
    """Given a k_dict from get_joints_at_kdist, extract observation vector.

    :param data: a structure containing the global state of the agent
    :param k_dict: the k_dict of an agent
    :param k_categories: the categories at every depth level
    :param global_dict: Not implemented for now
    :param global_categories: Not implemented for now
    :return:
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
                    if node.bodies is not None:
                        for body in node.bodies:
                            if category not in body_set_dict:
                                body_set_dict[category] = set()
                            if body not in body_set_dict[category]:
                                items = getattr(data, category)[body].tolist()
                                items = getattr(node, "body_fn", lambda _id, x: x)(
                                    body, items
                                )
                                obs_lst.extend(
                                    items if isinstance(items, list) else [items]
                                )
                                body_set_dict[category].add(body)

    # Add global observations
    body_set_dict = {}
    for category in global_categories:
        if category in ["qvel", "qpos", "qfrc_actuator"]:
            for joint in global_dict.get("joints", []):
                if category in joint.extra_obs:
                    items = joint.extra_obs[category](data).tolist()
                    obs_lst.extend(items if isinstance(items, list) else [items])
                elif category in ["qfrc_actuator"]:  # this is a "actuator forces" item
                    obs_lst.extend([data.qfrc_actuator[joint.qvel_ids]])
                else:
                    items = getattr(data, category)[
                        getattr(joint, "{}_ids".format(category))
                    ]
                    obs_lst.extend(items if isinstance(items, list) else [items])
        else:
            for body in global_dict.get("bodies", []):
                if category not in body_set_dict:
                    body_set_dict[category] = set()
                if body not in body_set_dict[category]:
                    obs_lst.extend(getattr(data, category)[body].tolist())
                    body_set_dict[category].add(body)

    return np.array(obs_lst)


def get_parts_and_edges(
    label: str, partitioning: str
) -> list[tuple[Node, ...], list[HyperEdge], dict[str : list[Node]]]:
    """
    :param label: the mujoco task to partition
    :param partitioning: the partioneing scheme
    :return:
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
        globals = {"joints": [root_x, root_y, root_z]}

        if partitioning is None:
            parts = [(bfoot, bshin, bthigh, ffoot, fshin, fthigh)]
        elif partitioning == "2x3":
            parts = [(bfoot, bshin, bthigh), (ffoot, fshin, fthigh)]
        elif partitioning == "6x1":
            parts = [(bfoot,), (bshin,), (bthigh,), (ffoot,), (fshin,), (fthigh,)]
        else:
            raise Exception("UNKNOWN partitioning config: {}".format(partitioning))

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

        hip1 = Node(
            "hip1",
            -8,
            -8,
            2,
            bodies=[torso, front_left_leg],
            body_fn=lambda _id, x: np.clip(x, -1, 1).tolist(),
        )
        ankle1 = Node(
            "ankle1",
            -7,
            -7,
            3,
            bodies=[front_left_leg, aux_1, ankle_1],
            body_fn=lambda _id, x: np.clip(x, -1, 1).tolist(),
        )
        hip2 = Node(
            "hip2",
            -6,
            -6,
            4,
            bodies=[torso, front_right_leg],
            body_fn=lambda _id, x: np.clip(x, -1, 1).tolist(),
        )
        ankle2 = Node(
            "ankle2",
            -5,
            -5,
            5,
            bodies=[front_right_leg, aux_2, ankle_2],
            body_fn=lambda _id, x: np.clip(x, -1, 1).tolist(),
        )
        hip3 = Node(
            "hip3",
            -4,
            -4,
            6,
            bodies=[torso, back_leg],
            body_fn=lambda _id, x: np.clip(x, -1, 1).tolist(),
        )
        ankle3 = Node(
            "ankle3",
            -3,
            -3,
            7,
            bodies=[back_leg, aux_3, ankle_3],
            body_fn=lambda _id, x: np.clip(x, -1, 1).tolist(),
        )
        hip4 = Node(
            "hip4",
            -2,
            -2,
            0,
            bodies=[torso, right_back_leg],
            body_fn=lambda _id, x: np.clip(x, -1, 1).tolist(),
        )
        ankle4 = Node(
            "ankle4",
            -1,
            -1,
            1,
            bodies=[right_back_leg, aux_4, ankle_4],
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
        globals = {"joints": [torso]}

        if partitioning is None:
            parts = [(hip1, ankle1, hip2, ankle2, hip3, ankle3, hip4, ankle4)]
        elif partitioning == "2x4":  # neighbouring legs together
            parts = [(hip1, ankle1, hip2, ankle2), (hip3, ankle3, hip4, ankle4)]
        elif partitioning == "2x4d":  # diagonal legs together
            parts = [(hip1, ankle1, hip3, ankle3), (hip2, ankle2, hip4, ankle4)]
        elif partitioning == "4x2":
            parts = [(hip1, ankle1), (hip2, ankle2), (hip3, ankle3), (hip4, ankle4)]
        else:
            raise Exception("UNKNOWN partitioning config: {}".format(partitioning))

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
        globals = {"joints": [root_x, root_y, root_z]}

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
            raise Exception("UNKNOWN partitioning config: {}".format(partitioning))

        return parts, edges, globals

    elif label in ["Humanoid-v4", "HumanoidStandup-v4"]:  # TODO
        # bodies
        torso = 0
        lwaist = 1
        pelvis = 2
        right_thigh = 3
        right_sin = 4
        right_foot = 5
        left_thigh = 6
        left_sin = 7
        left_foot = 8
        right_upper_arm = 9
        right_lower_arm = 10
        left_upper_arm = 11
        left_lower_arm = 12

        # define Mujoco-Graph
        abdomen_y = Node("abdomen_y", -16, -16, 0)
        abdomen_z = Node("abdomen_z", -17, -17, 1)
        abdomen_x = Node("abdomen_x", -15, -15, 2)
        right_hip_x = Node("right_hip_x", -14, -14, 3)
        right_hip_z = Node("right_hip_z", -13, -13, 4)
        right_hip_y = Node("right_hip_y", -12, -12, 5)
        right_knee = Node("right_knee", -11, -11, 6)
        left_hip_x = Node("left_hip_x", -10, -10, 7)
        left_hip_z = Node("left_hip_z", -9, -9, 8)
        left_hip_y = Node("left_hip_y", -8, -8, 9)
        left_knee = Node("left_knee", -7, -7, 10)
        right_shoulder1 = Node("right_shoulder1", -6, -6, 11)
        right_shoulder2 = Node("right_shoulder2", -5, -5, 12)
        right_elbow = Node("right_elbow", -4, -4, 13)
        left_shoulder1 = Node("left_shoulder1", -3, -3, 14)
        left_shoulder2 = Node("left_shoulder2", -2, -2, 15)
        left_elbow = Node("left_elbow", -1, -1, 16)

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
        globals = {"joints": [root]}

        if partitioning is None:
            parts = [
                (
                    left_shoulder1,
                    left_shoulder2,
                    abdomen_x,
                    abdomen_y,
                    abdomen_z,
                    right_shoulder1,
                    right_shoulder2,
                    right_elbow,
                    left_elbow,
                    left_hip_x,
                    left_hip_y,
                    left_hip_z,
                    right_hip_x,
                    right_hip_y,
                    right_hip_z,
                    right_knee,
                    left_knee,
                ),
            ]
        elif partitioning == "9|8":  # 17 in total
            # isolate upper and lower body
            parts = [
                (
                    left_shoulder1,
                    left_shoulder2,
                    abdomen_x,
                    abdomen_y,
                    abdomen_z,
                    right_shoulder1,
                    right_shoulder2,
                    right_elbow,
                    left_elbow,
                ),
                (
                    left_hip_x,
                    left_hip_y,
                    left_hip_z,
                    right_hip_x,
                    right_hip_y,
                    right_hip_z,
                    right_knee,
                    left_knee,
                ),
            ]
            # TODO: There could be tons of decompositions here

        else:
            raise Exception("UNKNOWN partitioning config: {}".format(partitioning))

        return parts, edges, globals

    elif label in ["Reacher-v4"]:

        # define Mujoco-Graph
        body0 = 1
        body1 = 2
        fingertip = 3
        joint0 = Node(
            "joint0",
            -4,
            -4,
            0,
            bodies=[body0, body1],
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
            bodies=[body1, fingertip],
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

        worldbody = 0
        target = 4
        target_x = Node(
            "target_x", -2, -2, None, extra_obs={"qvel": (lambda data: np.array([]))}
        )
        target_y = Node(
            "target_y", -1, -1, None, extra_obs={"qvel": (lambda data: np.array([]))}
        )
        globals = {"bodies": [worldbody, target], "joints": [target_x, target_y]}

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
            raise Exception("UNKNOWN partitioning config: {}".format(partitioning))

        return parts, edges, globals

    elif label in ["Swimmer-v4"]:
        # define Mujoco-Graph
        joint0 = Node(
            "rot2",
            -2,
            -2,
            0,
            extra_obs={
                "qvel": (lambda data: numpy.array([data.qvel[0], data.qvel[3]]))
            },
        )
        joint1 = Node(
            "rot3",
            -1,
            -1,
            1,
            extra_obs={
                "qvel": (lambda data: numpy.array([data.qvel[1], data.qvel[4]]))
            },
        )

        edges = [HyperEdge(joint0, joint1)]
        free_body_rot = Node("free_body_rot", 2, 2, None)
        globals = {"joints": [free_body_rot]}

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
            raise Exception("UNKNOWN partitioning config: {}".format(partitioning))

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
        globals = {"joints": [root_x, root_x, root_z]}

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
        elif partitioning == "2x3":
            # isolate upper and lower body
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
            raise Exception("UNKNOWN partitioning config: {}".format(partitioning))

        return parts, edges, globals

    elif label in ["coupled_half_cheetah-v4"]:

        # define Mujoco graph
        tendon = 0

        bthigh = Node(
            "bthigh",
            -6,
            -6,
            0,
            tendons=[tendon],
            extra_obs={
                "ten_J": lambda data: data.ten_J[tendon],
                "ten_length": lambda data: data.ten_length,
                "ten_velocity": lambda data: data.ten_velocity,
            },
        )
        bshin = Node("bshin", -5, -5, 1)
        bfoot = Node("bfoot", -4, -4, 2)
        fthigh = Node("fthigh", -3, -3, 3)
        fshin = Node("fshin", -2, -2, 4)
        ffoot = Node("ffoot", -1, -1, 5)

        bthigh2 = Node(
            "bthigh2",
            -6,
            -6,
            6,
            tendons=[tendon],
            extra_obs={
                "ten_J": lambda data: data.ten_J[tendon],
                "ten_length": lambda data: data.ten_length,
                "ten_velocity": lambda data: data.ten_velocity,
            },
        )
        bshin2 = Node("bshin2", -5, -5, 7)
        bfoot2 = Node("bfoot2", -4, -4, 8)
        fthigh2 = Node("fthigh2", -3, -3, 9)
        fshin2 = Node("fshin2", -2, -2, 10)
        ffoot2 = Node("ffoot2", -1, -1, 11)

        edges = [
            HyperEdge(bfoot, bshin),
            HyperEdge(bshin, bthigh),
            HyperEdge(bthigh, fthigh),
            HyperEdge(fthigh, fshin),
            HyperEdge(fshin, ffoot),
            HyperEdge(bfoot2, bshin2),
            HyperEdge(bshin2, bthigh2),
            HyperEdge(bthigh2, fthigh2),
            HyperEdge(fthigh2, fshin2),
            HyperEdge(fshin2, ffoot2),
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
        globals = {"joints": [root_x0, root_y0, root_z0, root_x1, root_y1, root_z1]}

        if partitioning is None:
            parts = [
                (
                    bfoot,
                    bshin,
                    bthigh,
                    ffoot,
                    fshin,
                    fthigh,
                    bfoot2,
                    bshin2,
                    bthigh2,
                    ffoot2,
                    fshin2,
                    fthigh2,
                ),
            ]
        elif partitioning == "1p1":
            parts = [
                (bfoot, bshin, bthigh, ffoot, fshin, fthigh),
                (bfoot2, bshin2, bthigh2, ffoot2, fshin2, fthigh2),
            ]
        else:
            raise Exception("UNKNOWN partitioning config: {}".format(partitioning))

        return parts, edges, globals

    elif label in ["manyagent_swimmer-v4"]:

        try:
            n_agents = int(partitioning.split("x")[0])
            n_segs_per_agents = int(partitioning.split("x")[1])
            n_segs = n_agents * n_segs_per_agents
        except Exception:
            raise Exception("UNKNOWN partitioning config: {}".format(partitioning))

        # Note: Default Swimmer corresponds to n_segs = 3

        # define Mujoco-Graph
        joints = [
            Node("rot{:d}".format(i), -n_segs + i, -n_segs + i, i)
            for i in range(0, n_segs)
        ]
        edges = [HyperEdge(joints[i], joints[i + 1]) for i in range(n_segs - 1)]
        globals = {}

        parts = [
            tuple(joints[i * n_segs_per_agents : (i + 1) * n_segs_per_agents])
            for i in range(n_agents)
        ]
        return parts, edges, globals

    elif label in ["manyagent_ant-v4"]:
        try:
            n_agents = int(partitioning.split("x")[0])
            n_segs_per_agents = int(partitioning.split("x")[1])
            n_segs = n_agents * n_segs_per_agents
        except Exception:
            raise Exception("UNKNOWN partitioning config: {}".format(partitioning))

        edges = []
        joints = []
        hip1m = None
        hip2m = None
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
                "hip1_{:d}".format(segment),
                -4 - off,
                -4 - off,
                2 + 4 * segment,
                bodies=[torso, front_right_leg],
                body_fn=lambda _id, x: np.clip(x, -1, 1).tolist(),
            )
            ankle1n = Node(
                "ankle1_{:d}".format(segment),
                -3 - off,
                -3 - off,
                3 + 4 * segment,
                bodies=[front_right_leg, aux1, ankle1],
                body_fn=lambda _id, x: np.clip(x, -1, 1).tolist(),
            )
            hip2n = Node(
                "hip2_{:d}".format(segment),
                -2 - off,
                -2 - off,
                0 + 4 * segment,
                bodies=[torso, back_leg],
                body_fn=lambda _id, x: np.clip(x, -1, 1).tolist(),
            )
            ankle2n = Node(
                "ankle2_{:d}".format(segment),
                -1 - off,
                -1 - off,
                1 + 4 * segment,
                bodies=[back_leg, aux2, ankle2],
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
        globals = {"joints": [free_joint]}

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

        return parts, edges, globals
    else:
        if partitioning is None:
            print("Warning: using single agent on unknown MuJoCo Environment: " + label)
            return tuple([tuple("0")]), None, None
        raise Exception("UNKNOWN label environment: {}".format(label))


def observation_structure(scenario: str) -> dict[str:int]:
    """
    :param scenario: the mujoco scenartio
    :return:
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
