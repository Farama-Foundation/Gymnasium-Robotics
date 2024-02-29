---
firstpage:
lastpage:
---

# Shadow Dexterous Hand

These environments are based on the [Shadow Dexterous Hand](https://www.shadowrobot.com/), 5 which is an anthropomorphic robotic hand with 24 degrees of freedom. Of those 24 joints, 20 can be controlled independently whereas the remaining ones are coupled joints.

* `HandReach-v2`: ShadowHand has to reach with its thumb and a selected finger until they meet at a desired goal position above the palm.
* `HandManipulateBlock-v1`: ShadowHand has to manipulate a block until it achieves a desired goal position and rotation.
* `HandManipulateEgg-v1`: ShadowHand has to manipulate an egg until it achieves a desired goal position and rotation.
* `HandManipulatePen-v1`: ShadowHand has to manipulate a pen until it achieves a desired goal position and rotation.

## Shadow Dexterous Hand with Touch Sensors

Touch sensor observations are also available in all Hand environments, with exception of `HandReach`. These environments add to the palm of the hand and the phalanges of the fingers 92 touch sensors with different recorded data depending on the environment. These touch sensors are:
- **Boolean Touch Sensor**: the observations of each touch sensor return a value of `0` if no contact is detected with and object, and `1` otherwise.
- **Continuous Touch Sensor**: the value returned by each touch sensor is a continuous value that represents the external force made by an object over the sensor.

These environments are instanceated by adding the following strings to the Hand environment id's: `_BooleanTouchSensor` or `_ContinuousTouchSensor`. For example, to add boolean touch sensors to `HandManipulateEgg-v1`, make the environment in the following way:

```python
import gymnasium as gym
import gymnasium_robotics

gym.register_envs(gymnasium_robotics)

env = gym.make('HandManipulateEgg_BooleanTouchSensors-v1')
```

```{raw} html
    :file: list.html
```

## References

If using the `Shadow Hand` environments, please cite:

```bibtex
@misc{1802.09464,
  Author = {Matthias Plappert and Marcin Andrychowicz and Alex Ray and Bob McGrew and Bowen Baker and Glenn Powell and Jonas Schneider and Josh Tobin and Maciek Chociej and Peter Welinder and Vikash Kumar and Wojciech Zaremba},
  Title = {Multi-Goal Reinforcement Learning: Challenging Robotics Environments and Request for Research},
  Year = {2018},
  Eprint = {arXiv:1802.09464},
}
```

To cite the `Shadow Dexterous Hand with Touch Sensors` environments, please use:

```bibtex
@article{melnik2021using,
  title={Using tactile sensing to improve the sample efficiency and performance of deep deterministic policy gradients for simulated in-hand manipulation tasks},
  author={Melnik, Andrew and Lach, Luca and Plappert, Matthias and Korthals, Timo and Haschke, Robert and Ritter, Helge},
  journal={Frontiers in Robotics and AI},
  pages={57},
  year={2021},
  publisher={Frontiers}
}
```

```{toctree}
:glob:
:hidden:
./*
```
