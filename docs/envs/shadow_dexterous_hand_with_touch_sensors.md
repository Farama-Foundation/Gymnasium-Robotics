---
firstpage:
lastpage:
---

# Shadow Dexterous Hand with Touch Sensors

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
hand/HandBlockTouchSensors
hand/HandEggTouchSensors
hand/HandPenTouchSensors

```
