# Fetch Reach

## Description

Fetch robot, simulated with MuJoCo has to move its end-effector to the desired goal position.


## Action Space

| Num | Action | Control Min | Control Max |
|--|--|--|--|
| 0 | Target displacement in the x direction | -1 | 1 |
| 1 | Target displacement in the y direction | -1 | 1 |
| 2 | Target displacement in the z direction | -1 | 1 |
| 3 | Unused | -1 | 1 |


## Observation Space

### Observation

| Num | Observation | Control Min | Control Max |
|--|--|--|--|
| 0 | x-coordinate of the gripper | -Inf | Inf |
| 1 | y-coordinate of the gripper | -Inf | Inf |
| 2 | z-coordinate of the gripper | -Inf | Inf |
| 3 | Half of the distance between the fingers | -Inf | Inf |
| 4 | Half of the distance between the fingers | -Inf | Inf |
| 5 | x-coordinate velocity of the gripper | -Inf | Inf |
| 6 | y-coordinate velocity of the gripper | -Inf | Inf |
| 7 | z-coordinate velocity of the gripper | -Inf | Inf |
| 8 | Left finger relative motion to the gripper | -Inf | Inf |
| 9 | Right finger relative motion to the gripper | -Inf | Inf |

3, 4, 8 and 9 remains very small since the gripper is blocked closed.


### Achieved and desired goal

| Num | Observation | Control Min | Control Max |
|--|--|--|--|
| 0 | x-coordinate of the gripper | -Inf | Inf |
| 1 | y-coordinate of the gripper | -Inf | Inf |
| 2 | z-coordinate of the gripper | -Inf | Inf |


## Rewards

The reward is 0.0 when the gripper is within 5 cm of the goal position. It is -1.0 otherwise.
