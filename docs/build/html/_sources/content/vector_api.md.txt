---
layout: "contents"
title: Vector API
---

# Vector API

## Vectorized Environments
*Vectorized environments* are environments that run multiple independent copies of the same environment in parallel using [multiprocessing](https://docs.python.org/3/library/multiprocessing.html). Vectorized environments take as input a batch of actions, and return a batch of observations. This is particularly useful, for example, when the policy is defined as a neural network that operates over a batch of observations.

Gym provides two types of vectorized environments:

- `gym.vector.SyncVectorEnv`, where the different copies of the environment are executed sequentially.
- `gym.vector.AsyncVectorEnv`, where the the different copies of the environment are executed in parallel using [multiprocessing](https://docs.python.org/3/library/multiprocessing.html). This creates one process per copy.


Similar to `gym.make`, you can run a vectorized version of a registered environment using the `gym.vector.make` function. This runs multiple copies of the same environment (in parallel, by default).

The following example runs 3 copies of the ``CartPole-v1`` environment in parallel, taking as input a vector of 3 binary actions (one for each copy of the environment), and returning an array of 3 observations stacked along the first dimension, with an array of rewards returned by each copy, and an array of booleans indicating if the episode in each parallel environment has ended.

```python
>>> envs = gym.vector.make("CartPole-v1", num_envs=3)
>>> envs.reset()
>>> actions = np.array([1, 0, 1])
>>> observations, rewards, dones, infos = envs.step(actions)

>>> observations
array([[ 0.00122802,  0.16228443,  0.02521779, -0.23700266],
        [ 0.00788269, -0.17490888,  0.03393489,  0.31735462],
        [ 0.04918966,  0.19421194,  0.02938497, -0.29495203]],
        dtype=float32)
>>> rewards
array([1., 1., 1.])
>>> dones
array([False, False, False])
>>> infos
({}, {}, {})
```

The function `gym.vector.make` is meant to be used only in basic cases (e.g. running multiple copies of the same registered environment). For any other use-cases, please use either the `SyncVectorEnv` for sequential execution, or `AsyncVectorEnv` for parallel execution. These use-cases may include:

- Running multiple instances of the same environment with different parameters (e.g. ``"Pendulum-v0"`` with different values for the gravity).
- Running multiple instances of an unregistered environment (e.g. a custom environment).
- Using a wrapper on some (but not all) environment copies.


### Creating a vectorized environment
To create a vectorized environment that runs multiple environment copies, you can wrap your parallel environments inside `gym.vector.SyncVectorEnv` (for sequential execution), or `gym.vector.AsyncVectorEnv` (for parallel execution, with [multiprocessing](https://docs.python.org/3/library/multiprocessing.html)). These vectorized environments take as input a list of callables specifying how the copies are created.

```python
>>> envs = gym.vector.AsyncVectorEnv([
...     lambda: gym.make("CartPole-v1"),
...     lambda: gym.make("CartPole-v1"),
...     lambda: gym.make("CartPole-v1")
... ])
```

Alternatively, to create a vectorized environment of multiple copies of the same registered environment, you can use the function `gym.vector.make()`.

```python
>>> envs = gym.vector.make("CartPole-v1", num_envs=3)  # Equivalent
```

To enable automatic batching of actions and observations, all of the environment copies must share the same `action_space` and `observation_space`. However, all of the parallel environments are not required to be exact copies of one another. For example, you can run 2 instances of ``Pendulum-v0`` with different values for gravity in a vectorized environment with:

```python
>>> env = gym.vector.AsyncVectorEnv([
...     lambda: gym.make("Pendulum-v0", g=9.81),
...     lambda: gym.make("Pendulum-v0", g=1.62)
... ])
```

See the `Observation & Action spaces` section for more information about automatic batching.

When using `AsyncVectorEnv` with either the ``spawn`` or ``forkserver`` start methods, you must wrap your code containing the vectorized environment with ``if __name__ == "__main__":``. See [this documentation](https://docs.python.org/3/library/multiprocessing.html#the-spawn-and-forkserver-start-methods) for more information.

```python
if __name__ == "__main__":
    envs = gym.vector.make("CartPole-v1", num_envs=3, context="spawn")
```
### Working with vectorized environments
While standard Gym environments take a single action and return a single observation (with a reward, and boolean indicating termination), vectorized environments take a *batch of actions* as input, and return a *batch of observations*, together with an array of rewards and booleans indicating if the episode ended in each environment copy.


```python
>>> envs = gym.vector.make("CartPole-v1", num_envs=3)
>>> envs.reset()
array([[ 0.00198895, -0.00569421, -0.03170966,  0.00126465],
       [-0.02658334,  0.00755256,  0.04376719, -0.00266695],
       [-0.02898625,  0.04779156,  0.02686412, -0.01298284]],
      dtype=float32)

>>> actions = np.array([1, 0, 1])
>>> observations, rewards, dones, infos = envs.step(actions)

>>> observations
array([[ 0.00187507,  0.18986781, -0.03168437, -0.301252  ],
       [-0.02643229, -0.18816885,  0.04371385,  0.3034975 ],
       [-0.02803041,  0.24251814,  0.02660446, -0.29707024]],
      dtype=float32)
>>> rewards
array([1., 1., 1.])
>>> dones
array([False, False, False])
>>> infos
({}, {}, {})
```

Vectorized environments are compatible with any environment, regardless of the action and observation spaces (e.g. container spaces like `gym.spaces.Dict`, or any arbitrarily nested spaces). In particular, vectorized environments can automatically batch the observations returned by `VectorEnv.reset` and `VectorEnv.step` for any standard Gym `Space` (e.g. `gym.spaces.Box`, `gym.spaces.Discrete`, `gym.spaces.Dict`, or any nested structure thereof). Similarly, vectorized environments can take batches of actions from any standard Gym `Space`.

```python
>>> class DictEnv(gym.Env):
...     observation_space = gym.spaces.Dict({
...         "position": gym.spaces.Box(-1., 1., (3,), np.float32),
...         "velocity": gym.spaces.Box(-1., 1., (2,), np.float32)
...     })
...     action_space = gym.spaces.Dict({
...         "fire": gym.spaces.Discrete(2),
...         "jump": gym.spaces.Discrete(2),
...         "acceleration": gym.spaces.Box(-1., 1., (2,), np.float32)
...     })
...
...     def reset(self):
...         return self.observation_space.sample()
...
...     def step(self, action):
...         observation = self.observation_space.sample()
...         return (observation, 0., False, {})

>>> envs = gym.vector.AsyncVectorEnv([lambda: DictEnv()] * 3)
>>> envs.observation_space
Dict(position:Box(-1.0, 1.0, (3, 3), float32), velocity:Box(-1.0, 1.0, (3, 2), float32))
>>> envs.action_space
Dict(fire:MultiDiscrete([2 2 2]), jump:MultiDiscrete([2 2 2]), acceleration:Box(-1.0, 1.0, (3, 2), float32))

>>> envs.reset()
>>> actions = {
...     "fire": np.array([1, 1, 0]),
...     "jump": np.array([0, 1, 0]),
...     "acceleration": np.random.uniform(-1., 1., size=(3, 2))
... }
>>> observations, rewards, dones, infos = envs.step(actions)
>>> observations
{"position": array([[-0.5337036 ,  0.7439302 ,  0.41748118],
                    [ 0.9373266 , -0.5780453 ,  0.8987405 ],
                    [-0.917269  , -0.5888639 ,  0.812942  ]], dtype=float32),
"velocity": array([[ 0.23626241, -0.0616814 ],
                   [-0.4057572 , -0.4875375 ],
                   [ 0.26341468,  0.72282314]], dtype=float32)}
```

The environment copies inside a vectorized environment automatically call `gym.Env.reset` at the end of an episode. In the following example, the episode of the 3rd copy ends after 2 steps (the agent fell in a hole), and the paralle environment gets reset (observation ``0``).

```python
>>> envs = gym.vector.make("FrozenLake-v1", num_envs=3, is_slippery=False)
>>> envs.reset()
array([0, 0, 0])
>>> observations, rewards, dones, infos = envs.step(np.array([1, 2, 2]))
>>> observations, rewards, dones, infos = envs.step(np.array([1, 2, 1]))

>>> dones
array([False, False,  True])
>>> observations
array([8, 2, 0])
```

## Observation & Action spaces
Like any Gym environment, vectorized environments contain the two properties `VectorEnv.observation_space` and `VectorEnv.action_space` to specify the observation and action spaces of the environments. Since vectorized environments operate on multiple environment copies, where the actions taken and observations returned by all of the copies are batched together, the observation and action *spaces* are batched as well so that the input actions are valid elements of `VectorEnv.action_space`, and the observations are valid elements of `VectorEnv.observation_space`.

```python
>>> envs = gym.vector.make("CartPole-v1", num_envs=3)
>>> envs.observation_space
Box([[-4.8 ...]], [[4.8 ...]], (3, 4), float32)
>>> envs.action_space
MultiDiscrete([2 2 2])
```

In order to appropriately batch the observations and actions in vectorized environments, the observation and action spaces of all of the copies are required to be identical.

```python
>>> envs = gym.vector.AsyncVectorEnv([
...     lambda: gym.make("CartPole-v1"),
...     lambda: gym.make("MountainCar-v0")
... ])
RuntimeError: Some environments have an observation space different from `Box([-4.8 ...], [4.8 ...], (4,), float32)`. 
In order to batch observations, the observation spaces from all environments must be equal.
```
However, sometimes it may be handy to have access to the observation and action spaces of a particular copy, and not the batched spaces. You can access those with the properties `VectorEnv.single_observation_space` and `VectorEnv.single_action_space` of the vectorized environment.

```python
>>> envs = gym.vector.make("CartPole-v1", num_envs=3)
>>> envs.single_observation_space
Box([-4.8 ...], [4.8 ...], (4,), float32)
>>> envs.single_action_space
Discrete(2)
```
This is convenient, for example, if you instantiate a policy. In the following example, we use `VectorEnv.single_observation_space` and `VectorEnv.single_action_space` to define the weights of a linear policy. Note that, thanks to the vectorized environment, we can apply the policy directly to the whole batch of observations with a single call to `policy`.

```python
>>> from gym.spaces.utils import flatdim
>>> from scipy.special import softmax

>>> def policy(weights, observations):
...     logits = np.dot(observations, weights)
...     return softmax(logits, axis=1)

>>> envs = gym.vector.make("CartPole-v1", num_envs=3)
>>> weights = np.random.randn(
...     flatdim(envs.single_observation_space),
...     envs.single_action_space.n
... )
>>> observations = envs.reset()
>>> actions = policy(weights, observations).argmax(axis=1)
>>> observations, rewards, dones, infos = envs.step(actions)
```

## Intermediate Usage

### Shared memory
`AsyncVectorEnv` runs each environment copy inside an individual process. At each call to `AsyncVectorEnv.reset` or `AsyncVectorEnv.step`, the observations of all of the parallel environments are sent back to the main process. To avoid expensive transfers of data between processes, especially with large observations (e.g. images), `AsyncVectorEnv` uses a shared memory by default (``shared_memory=True``) that processes can write to and read from at minimal cost. This can increase the throughout of the vectorized environment.

```python
>>> env_fns = [lambda: gym.make("BreakoutNoFrameskip-v4")] * 5

>>> envs = gym.vector.AsyncVectorEnv(env_fns, shared_memory=False)
>>> envs.reset()
>>> %timeit envs.step(envs.action_space.sample())
2.23 ms ± 136 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

>>> envs = gym.vector.AsyncVectorEnv(env_fns, shared_memory=True)
>>> envs.reset()
>>> %timeit envs.step(envs.action_space.sample())
1.36 ms ± 15.4 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
```

### Exception handling
Because sometimes things may not go as planned, the exceptions raised in any given environment copy are re-raised in the vectorized environment, even when the copy run in parallel with `AsyncVectorEnv`. This way, you can choose how to handle these exceptions yourself (with ``try ... except``).

```python
>>> class ErrorEnv(gym.Env):
...     observation_space = gym.spaces.Box(-1., 1., (2,), np.float32)
...     action_space = gym.spaces.Discrete(2)
...
...     def reset(self):
...         return np.zeros((2,), dtype=np.float32)
...
...     def step(self, action):
...         if action == 1:
...             raise ValueError("An error occurred.")
...         observation = self.observation_space.sample()
...         return (observation, 0., False, {})

>>> envs = gym.vector.AsyncVectorEnv([lambda: ErrorEnv()] * 3)
>>> observations = envs.reset()
>>> observations, rewards, dones, infos = envs.step(np.array([0, 0, 1]))
ERROR: Received the following error from Worker-2: ValueError: An error occurred.
ERROR: Shutting down Worker-2.
ERROR: Raising the last exception back to the main process.
ValueError: An error occurred.
```

## Advanced Usage

### Custom spaces
Vectorized environments will batch actions and observations if they are elements from standard Gym spaces, such as `gym.spaces.Box`, `gym.spaces.Discrete`, or `gym.spaces.Dict`. However, if you create your own environment with a custom action and/or observation space (inheriting from `gym.Space`), the vectorized environment will not attempt to automatically batch the actions/observations, and instead it will return the raw tuple of elements from all parallel environments.

In the following example, we create a new environment `SMILESEnv`, whose observations are strings representing the [SMILES](https://en.wikipedia.org/wiki/Simplified_molecular-input_line-entry_system) notation of a molecular structure, with a custom observation space `SMILES`. The observations returned by the vectorized environment are contained in a tuple of strings. 

```python
>>> class SMILES(gym.Space):
...     def __init__(self, symbols):
...         super().__init__()
...         self.symbols = symbols
...
...     def __eq__(self, other):
...         return self.symbols == other.symbols

>>> class SMILESEnv(gym.Env):
...     observation_space = SMILES("][()CO=")
...     action_space = gym.spaces.Discrete(7)
...
...     def reset(self):
...         self._state = "["
...         return self._state
...
...     def step(self, action):
...         self._state += self.observation_space.symbols[action]
...         reward = done = (action == 0)
...         return (self._state, float(reward), done, {})

>>> envs = gym.vector.AsyncVectorEnv(
...     [lambda: SMILESEnv()] * 3,
...     shared_memory=False
... )
>>> envs.reset()
>>> observations, rewards, dones, infos = envs.step(np.array([2, 5, 4]))
>>> observations
('[(', '[O', '[C')
```

Custom observation and action spaces may inherit from the `gym.Space` class. However, most use-cases should be covered by the existing space classes (e.g. `gym.spaces.Box`, `gym.spaces.Discrete`, etc...), and container classes (`gym.spaces.Tuple` and `gym.spaces.Dict`). Moreover, some implementations of reinforcement learning algorithms might not handle custom spaces properly. Use custom spaces with care.

If you use `AsyncVectorEnv` with a custom observation space, you must set ``shared_memory=False``, since shared memory and automatic batching is not compatible with custom spaces. In general if you use custom spaces with `AsyncVectorEnv`, the elements of those spaces must be `pickleable`.


## API Reference

### VectorEnv

```{eval-rst}
.. attribute:: gym.vector.VectorEnv.action_space

    The (batched) action space. The input actions of `step` must be valid elements of `action_space`.::

        >>> envs = gym.vector.make("CartPole-v1", num_envs=3)
        >>> envs.action_space
        MultiDiscrete([2 2 2])

.. attribute:: gym.vector.VectorEnv.observation_space

    The (batched) observation space. The observations returned by `reset` and `step` are valid elements of `observation_space`.::

        >>> envs = gym.vector.make("CartPole-v1", num_envs=3)
        >>> envs.observation_space
        Box([[-4.8 ...]], [[4.8 ...]], (3, 4), float32)

.. attribute:: gym.vector.VectorEnv.single_action_space

    The action space of an environment copy.::

        >>> envs = gym.vector.make("CartPole-v1", num_envs=3)
        >>> envs.single_action_space
        Discrete(2)

.. attribute:: gym.vector.VectorEnv.single_observation_space

    The observation space of an environment copy.::

        >>> envs = gym.vector.make("CartPole-v1", num_envs=3)
        >>> envs.single_action_space
        Box([-4.8 ...], [4.8 ...], (4,), float32)
``` 



### Reset

```{eval-rst}
.. automethod:: gym.vector.VectorEnv.reset
``` 

```python
>>> envs = gym.vector.make("CartPole-v1", num_envs=3)
>>> envs.reset()
array([[-0.04456399,  0.04653909,  0.01326909, -0.02099827],
        [ 0.03073904,  0.00145001, -0.03088818, -0.03131252],
        [ 0.03468829,  0.01500225,  0.01230312,  0.01825218]],
        dtype=float32)
```
### Step

```{eval-rst}
.. automethod:: gym.vector.VectorEnv.step
``` 

```python
>>> envs = gym.vector.make("CartPole-v1", num_envs=3)
>>> envs.reset()
>>> actions = np.array([1, 0, 1])
>>> observations, rewards, dones, infos = envs.step(actions)

>>> observations
array([[ 0.00122802,  0.16228443,  0.02521779, -0.23700266],
        [ 0.00788269, -0.17490888,  0.03393489,  0.31735462],
        [ 0.04918966,  0.19421194,  0.02938497, -0.29495203]],
        dtype=float32)
>>> rewards
array([1., 1., 1.])
>>> dones
array([False, False, False])
>>> infos
({}, {}, {})
```

### Seed

```{eval-rst}
.. automethod:: gym.vector.VectorEnv.seed
``` 

```python
>>> envs = gym.vector.make("CartPole-v1", num_envs=3)
>>> envs.seed([1, 3, 5])
>>> envs.reset()
array([[ 0.03073904,  0.00145001, -0.03088818, -0.03131252],
        [ 0.02281231, -0.02475473,  0.02306162,  0.02072129],
        [-0.03742824, -0.02316945,  0.0148571 ,  0.0296055 ]],
        dtype=float32)
```
