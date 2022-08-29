import copy
import os
from typing import Optional, Union

import gym
import numpy as np
from gym import error, logger, spaces
from gym.utils.renderer import Renderer

from gym_robotics import GoalEnv

MUJOCO_PY_NOT_INSTALLED = False
MUJOCO_NOT_INSTALLED = False

try:
    import mujoco_py

    from gym_robotics.utils import mujoco_py_utils
except ImportError as e:
    MUJOCO_PY_IMPORT_ERROR = e
    MUJOCO_PY_NOT_INSTALLED = True

try:
    import mujoco

    from gym_robotics.utils import mujoco_utils
except ImportError as e:
    MUJOCO_IMPORT_ERROR = e
    MUJOCO_NOT_INSTALLED = True


DEFAULT_SIZE = 480


class BaseRobotEnv(GoalEnv):

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "single_rgb_array",
        ],
        "render_fps": 60,
    }

    def __init__(
        self,
        model_path: str,
        initial_qpos,
        n_actions: int,
        n_substeps: int,
        render_mode: Optional[str] = None,
        width: int = DEFAULT_SIZE,
        height: int = DEFAULT_SIZE,
    ):

        if model_path.startswith("/"):
            self.fullpath = model_path
        else:
            self.fullpath = os.path.join(
                os.path.dirname(__file__), "assets", model_path
            )
        if not os.path.exists(self.fullpath):
            raise OSError(f"File {self.fullpath} does not exist")

        self.n_substeps = n_substeps

        self.initial_qpos = initial_qpos

        self.width = width
        self.height = height
        self._initialize_simulation()

        self.viewer = None
        self._viewers = {}

        self.goal = np.zeros(0)
        obs = self._get_obs()

        self.action_space = spaces.Box(-1.0, 1.0, shape=(n_actions,), dtype="float32")
        self.observation_space = spaces.Dict(
            dict(
                desired_goal=spaces.Box(
                    -np.inf, np.inf, shape=obs["achieved_goal"].shape, dtype="float64"
                ),
                achieved_goal=spaces.Box(
                    -np.inf, np.inf, shape=obs["achieved_goal"].shape, dtype="float64"
                ),
                observation=spaces.Box(
                    -np.inf, np.inf, shape=obs["observation"].shape, dtype="float64"
                ),
            )
        )

        self.render_mode = render_mode

        self.renderer = Renderer(self.render_mode, self._render)

    # Env methods
    # ----------------------------

    def step(self, action):
        if np.array(action).shape != self.action_space.shape:
            raise ValueError("Action dimension mismatch")

        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action)

        self._mujoco_step(action)

        self._step_callback()
        obs = self._get_obs()

        info = {
            "is_success": self._is_success(obs["achieved_goal"], self.goal),
        }

        terminated = False
        truncated = False

        reward = self.compute_reward(obs["achieved_goal"], self.goal, info)
        self.renderer.render_step()

        return obs, reward, terminated, truncated, info

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ):
        # Attempt to reset the simulator. Since we randomize initial conditions, it
        # is possible to get into a state with numerical issues (e.g. due to penetration or
        # Gimbel lock) or we may not achieve an initial condition (e.g. an object is within the hand).
        # In this case, we just keep randomizing until we eventually achieve a valid initial
        # configuration.
        super().reset(seed=seed)
        did_reset_sim = False
        while not did_reset_sim:
            did_reset_sim = self._reset_sim()
        self.goal = self._sample_goal().copy()
        obs = self._get_obs()
        self.renderer.reset()
        self.renderer.render_step()
        if not return_info:
            return obs
        else:
            return obs, {}

    def close(self):
        if self.viewer is not None:
            self.viewer = None
            self._viewers = {}

    def _render(
        self,
        mode: str = "human",
        camera_id: Optional[int] = None,
        camera_name: Optional[str] = None,
    ):
        """
        Render a frame from the MuJoCo simulation as specified by the render_mode.
        """
        raise NotImplementedError

    def render(self):
        self._render_callback()

        return self.renderer.get_renders()

    # Extension methods
    # ----------------------------
    def _mujoco_step(self, action):
        raise NotImplementedError

    def _get_viewer(self, mode):
        raise NotImplementedError

    def _reset_sim(self):
        """Resets a simulation and indicates whether or not it was successful.
        If a reset was unsuccessful (e.g. if a randomized state caused an error in the
        simulation), this method should indicate such a failure by returning False.
        In such a case, this method will be called again to attempt a the reset again.
        """
        return True

    def _initialize_simulation(self):
        """
        Initialize MuJoCo simulation data structures mjModel and mjData.
        """
        raise NotImplementedError

    def _get_obs(self):
        """Returns the observation."""
        raise NotImplementedError()

    def _set_action(self, action):
        """Applies the given action to the simulation."""
        raise NotImplementedError()

    def _is_success(self, achieved_goal, desired_goal):
        """Indicates whether or not the achieved goal successfully achieved the desired goal."""
        raise NotImplementedError()

    def _sample_goal(self):
        """Samples a new goal and returns it."""
        raise NotImplementedError()

    def _env_setup(self, initial_qpos):
        """Initial configuration of the environment. Can be used to configure initial state
        and extract information from the simulation.
        """
        pass

    def _viewer_setup(self):
        """Initial configuration of the viewer. Can be used to set the camera position,
        for example.
        """
        pass

    def _render_callback(self):
        """A custom callback that is called before rendering. Can be used
        to implement custom visualizations.
        """
        pass

    def _step_callback(self):
        """A custom callback that is called after stepping the simulation. Can be used
        to enforce additional constraints on the simulation state.
        """
        pass


class MujocoRobotEnv(BaseRobotEnv):
    def __init__(self, **kwargs):
        self._mujoco = mujoco
        self._utils = mujoco_utils

        super().__init__(**kwargs)

    def _initialize_simulation(self):
        self.model = self._mujoco.MjModel.from_xml_path(self.fullpath)
        self.data = self._mujoco.MjData(self.model)
        self._model_names = self._utils.MujocoModelNames(self.model)

        self.model.vis.global_.offwidth = self.width
        self.model.vis.global_.offheight = self.height

        self._env_setup(initial_qpos=self.initial_qpos)
        self.initial_time = self.data.time
        self.initial_qpos = np.copy(self.data.qpos)
        self.initial_qvel = np.copy(self.data.qvel)

    def _reset_sim(self):
        self.data.time = self.initial_time
        self.data.qpos[:] = np.copy(self.initial_qpos)
        self.data.qvel[:] = np.copy(self.initial_qvel)
        if self.model.na != 0:
            self.data.act[:] = None

        mujoco.mj_forward(self.model, self.data)
        return super()._reset_sim()

    def _render(self, mode: str = "human"):
        assert mode in self.metadata["render_modes"]

        if mode in {
            "rgb_array",
            "single_rgb_array",
        }:
            self._get_viewer(mode).render(height=480, width=480)
            data = self._get_viewer(mode).read_pixels(
                height=480, width=480, depth=False
            )
            # original image is upside-down, so flip it
            return data[::-1, :, :]
        elif mode == "human":
            self._get_viewer(mode).render()

    def _get_viewer(
        self, mode
    ) -> Union["gym.envs.mujoco.Viewer", "gym.envs.mujoco.RenderContextOffscreen"]:
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == "human":
                from gym.envs.mujoco.mujoco_rendering import Viewer

                self.viewer = Viewer(self.model, self.data)
            elif mode in {
                "rgb_array",
                "single_rgb_array",
            }:
                from gym.envs.mujoco.mujoco_rendering import RenderContextOffscreen

                self.viewer = RenderContextOffscreen(
                    model=self.model, data=self.data, width=480, height=480
                )
            self._viewer_setup()
            self._viewers[mode] = self.viewer
        return self.viewer

    @property
    def dt(self):
        return self.model.opt.timestep * self.n_substeps

    def _mujoco_step(self, action):
        self._mujoco.mj_step(self.model, self.data, nstep=self.n_substeps)


class MujocoPyRobotEnv(BaseRobotEnv):
    def __init__(self, **kwargs):
        if MUJOCO_PY_NOT_INSTALLED:
            raise error.DependencyNotInstalled(
                f"{MUJOCO_PY_IMPORT_ERROR}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)"
            )
        self._mujoco_py = mujoco_py
        self._utils = mujoco_py_utils

        logger.warn(
            "This version of the mujoco environments depends "
            "on the mujoco-py bindings, which are no longer maintained "
            "and may stop working. Please upgrade to the v4 versions of "
            "the environments (which depend on the mujoco python bindings instead), unless "
            "you are trying to precisely replicate previous works)."
        )

        super().__init__(**kwargs)

    def _initialize_simulation(self):
        self.model = self._mujoco_py.load_model_from_path(self.fullpath)
        self.sim = self._mujoco_py.MjSim(self.model, nsubsteps=self.n_substeps)
        self.data = self.sim.data

        self._env_setup(initial_qpos=self.initial_qpos)
        self.initial_state = copy.deepcopy(self.sim.get_state())

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)
        self.sim.forward()
        return super()._reset_sim()

    def _render(self, mode: str = "human"):
        width, height = self.width, self.height
        assert mode in self.metadata["render_modes"]
        if mode in {
            "rgb_array",
            "single_rgb_array",
        }:
            self._get_viewer(mode).render(width, height)
            # window size used for old mujoco-py:
            data = self._get_viewer(mode).read_pixels(width, height, depth=False)
            # original image is upside-down, so flip it
            return data[::-1, :, :]
        elif mode == "human":
            self._get_viewer(mode).render()

    def _get_viewer(
        self, mode
    ) -> Union["mujoco_py.MjViewer", "mujoco_py.MjRenderContextOffscreen"]:
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == "human":
                self.viewer = self._mujoco_py.MjViewer(self.sim)

            elif mode in {
                "rgb_array",
                "depth_array",
                "single_rgb_array",
                "single_depth_array",
            }:
                self.viewer = self._mujoco_py.MjRenderContextOffscreen(self.sim, -1)
            self._viewer_setup()
            self._viewers[mode] = self.viewer
        return self.viewer

    @property
    def dt(self):
        return self.sim.model.opt.timestep * self.sim.nsubsteps

    def _mujoco_step(self, action):
        self.sim.step()
