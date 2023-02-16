import gymnasium as gym


class SetInitialState(gym.Wrapper, gym.utils.RecordConstructorArgs):
    def __init__(self, env):
        gym.utils.RecordConstructorArgs.__init__(self)
        gym.Wrapper.__init__(self, env)

    def reset(self, initial_state_dict=None, *args, **kwargs):
        result = self.env.reset(*args, **kwargs)
        if initial_state_dict is not None:
            self.env.set_env_state(initial_state_dict)
        return result