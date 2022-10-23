---
firstpage:
lastpage:
---

## Fetch

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
