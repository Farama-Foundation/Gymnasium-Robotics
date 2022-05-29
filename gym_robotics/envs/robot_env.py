import os
import copy
from typing import Optional

import numpy as np

import gym
from gym import error, logger, spaces
from gym.utils import seeding

from gym_robotics import GoalEnv

try:
    import mujoco_py
except ImportError as e:
    raise error.DependencyNotInstalled(
        "{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(
            e
        )
    )

DEFAULT_SIZE = 500


class RobotEnv(GoalEnv):
    def __init__(self, model_path, initial_qpos, n_actions, n_substeps, mujoco_bindings="mujoco"):


        if model_path.startswith("/"):
            fullpath = model_path
        else:
            fullpath = os.path.join(os.path.dirname(__file__), "assets", model_path)
        if not os.path.exists(fullpath):
            raise OSError(f"File {fullpath} does not exist")

        self.n_substeps = n_substeps
        
        if mujoco_bindings == "mujoco_py":
            logger.warn(
                "This version of the mujoco environments depends "
                "on the mujoco-py bindings, which are no longer maintained "
                "and may stop working. Please upgrade to the v4 versions of "
                "the environments (which depend on the mujoco python bindings instead), unless "
                "you are trying to precisely replicate previous works)."
            )
            try:
                import mujoco_py

                self._mujoco_bindings = mujoco_py

            except ImportError as e:
                raise error.DependencyNotInstalled(
                    "{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(
                        e
                    )
                )

            self.model = self._mujoco_bindings.load_model_from_path(fullpath)
            self.sim = self._mujoco_bindings.MjSim(self.model, nsubsteps=n_substeps)
            self.data = self.sim.data

            self._env_setup(initial_qpos=initial_qpos)
            self.initial_state = copy.deepcopy(self.sim.get_state())

        elif mujoco_bindings == "mujoco":
            try:
                import mujoco

                self._mujoco_bindings = mujoco

            except ImportError as e:
                raise error.DependencyNotInstalled(
                    f"{e}. (HINT: you need to install mujoco)"
                )
            self.model = self._mujoco_bindings.MjModel.from_xml_path(fullpath)
            self.data = self._mujoco_bindings.MjData(self.model)

            self._env_setup(initial_qpos=initial_qpos)
            self.initial_time = self.data.time
            self.initial_qpos = np.copy(self.data.qpos)
            self.initial_qvel = np.copy(self.data.qvel)
            

    
        self.viewer = None
        self._viewers = {}

        self.metadata = {
            "render.modes": ["human", "rgb_array"],
            "video.frames_per_second": int(np.round(1.0 / self.dt)),
        }

        

        self.goal = np.zeros(0)
        obs = self._get_obs()
        self.action_space = spaces.Box(-1.0, 1.0, shape=(n_actions,), dtype="float32")
        self.observation_space = spaces.Dict(
            dict(
                desired_goal=spaces.Box(
                    -np.inf, np.inf, shape=obs["achieved_goal"].shape, dtype="float32"
                ),
                achieved_goal=spaces.Box(
                    -np.inf, np.inf, shape=obs["achieved_goal"].shape, dtype="float32"
                ),
                observation=spaces.Box(
                    -np.inf, np.inf, shape=obs["observation"].shape, dtype="float32"
                ),
            )
        )


    @property
    def dt(self):
        if self._mujoco_bindings.__name__ == "mujoco_py":    
            return self.sim.model.opt.timestep * self.sim.nsubsteps
        else:
            return self.model.opt.timestep * self.n_substeps

    # Env methods
    # ----------------------------

    def step(self, action):
        if np.array(action).shape != self.action_space.shape:
            raise ValueError("Action dimension mismatch")

        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action)
        if self._mujoco_bindings.__name__ == "mujoco_py":
                self.sim.step()
        else:
            for _ in range(self.n_substeps):
                self._mujoco_bindings.mj_step(self.model, self.data)
        
        self._step_callback()
        obs = self._get_obs()

        done = False
        info = {
            "is_success": self._is_success(obs["achieved_goal"], self.goal),
        }
        reward = self.compute_reward(obs["achieved_goal"], self.goal, info)
        return obs, reward, done, info

    def reset(self, seed: Optional[int] = None):
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
        return obs

    def close(self):
        if self.viewer is not None:
            self.viewer = None
            self._viewers = {}

    def render(self, mode="human", width=DEFAULT_SIZE, height=DEFAULT_SIZE):
        self._render_callback()
        if mode == "rgb_array":
            self._get_viewer(mode).render(width, height)
            # window size used for old mujoco-py:
            data = self._get_viewer(mode).read_pixels(width, height, depth=False)
            # original image is upside-down, so flip it
            return data[::-1, :, :]
        elif mode == "human":
            self._get_viewer(mode).render()

    def _get_viewer(self, mode, width=DEFAULT_SIZE, height=DEFAULT_SIZE):
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == "human":
                if self._mujoco_bindings.__name__ == "mujoco_py":
                    self.viewer = self._mujoco_bindings.MjViewer(self.sim)
                else:
                    from gym.envs.mujoco.mujoco_rendering import Viewer

                    self.viewer = Viewer(self.model, self.data)
            elif mode == "rgb_array":
                if self._mujoco_bindings.__name__ == "mujoco_py":
                    self.viewer = self._mujoco_bindings.MjRenderContextOffscreen(
                        self.sim, -1
                    )
                else:
                    from gym.envs.mujoco.mujoco_rendering import RenderContextOffscreen

                    self.viewer = RenderContextOffscreen(
                        width, height, self.model, self.data
                    )
            self._viewer_setup()
            self._viewers[mode] = self.viewer
        return self.viewer

    # Extension methods
    # ----------------------------

    def _reset_sim(self):
        """Resets a simulation and indicates whether or not it was successful.
        If a reset was unsuccessful (e.g. if a randomized state caused an error in the
        simulation), this method should indicate such a failure by returning False.
        In such a case, this method will be called again to attempt a the reset again.
        """
        if self._mujoco_bindings.__name__ == "mujoco_py":
            self.sim.set_state(self.initial_state)
            self.sim.forward()
        else:
            self.data.time = self.initial_time
            self.data.qpos[:] = np.copy(self.initial_qpos)
            self.data.qvel[:] = np.copy(self.initial_qvel)
            if self.model.na != 0:
                self.data.act[:] = None
            
            self._mujoco_bindings.mj_forward(self.model, self.data)
        
        return True

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
