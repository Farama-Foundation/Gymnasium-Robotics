---
firstpage:
lastpage:
---

## Goal-Aware Environments

```{eval-rst}
.. autoclass:: gymnasium_robotics.core.GoalEnv
```

### Methods
```{eval-rst}
.. gymnasium_robotics.core.GoalEnv.compute_reward
.. gymnasium_robotics.core.GoalEnv.compute_terminated
.. gymnasium_robotics.core.GoalEnv.compute_truncated
```

## Fetch environments

The Fetch environments are based on the 7-DoF [Fetch Mobile Manipulator](https://fetchrobotics.com/) arm, with a two-fingered parallel gripper attached to it. The main environment tasks are the following: 

* `FetchReach-v2`: Fetch has to move its end-effector to the desired goal position.
* `FetchPush-v2`: Fetch has to move a box by pushing it until it reaches a desired goal position.
* `FetchSlide-v2`: Fetch has to hit a puck across a long table such that it slides and comes to rest on the desired goal.
* `FetchPickAndPlace-v2`: Fetch has to pick up a box from a table using its gripper and move it to a desired goal above the table.


```{toctree}
:hidden:
fetch/FetchReach
fetch/FetchSlide
fetch/FetchPickAndPlace
fetch/FetchPush

```


## Shadow Dexterous Hand environments

These environments are based on the [Shadow Dexterous Hand](https://www.shadowrobot.com/), 5 which is an anthropomorphic robotic hand with 24 degrees of freedom. Of those 24 joints, 20 can be can be controlled independently whereas the remaining ones are coupled joints.

* `HandReach-v1`: ShadowHand has to reach with its thumb and a selected finger until they meet at a desired goal position above the palm.
* `HandManipulateBlock-v1`: ShadowHand has to manipulate a block until it achieves a desired goal position and rotation.
* `HandManipulateEgg-v1`: ShadowHand has to manipulate an egg until it achieves a desired goal position and rotation.
* `HandManipulatePen-v1`: ShadowHand has to manipulate a pen until it achieves a desired goal position and rotation.


```{toctree}
:hidden:
hand/HandReach
hand/HandBlock
hand/HandEgg
hand/HandPen

```


# Hand environments with Touch Sensors

Touch sensor observations are also available in all Hand environments, with exception of `HandReach`. These environments add to the palm of the hand and the phalanges of the fingers 92 touch sensors with different recorded data depending on the environment. These touch sensors are:
- **Boolean Touch Sensor**: the observations of each touch sensor return a value of `0` if no contact is detected with and object, and `1` otherwise.
- **Continuous Touch Sensor**: the value returned by each touch sensor is a continuous value that represents the external force made by an object over the sensor.

These environments are instanceated by adding the following strings to the Hand environment id's: `_BooleanTouchSensor` or `_ContinuousTouchSensor`. For example, to add boolean touch sensors to `HandManipulateEgg-v1`, make the environment in the following way:

```python
import gymnasium as gym

env = gym.make('HandManipulateEgg_BooleanTouchSensor-v1')
```

```{toctree}
:hidden:
hand/ManipulateTouchSensors
hand/HandBlockTouchSensors
hand/HandEggTouchSensors
hand/HandPenTouchSensors

```
