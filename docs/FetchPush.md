# Fetch Push

## Description

Fetch robot, simulated with MuJoCo has to move a cube to the desired goal position.

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
| 3 | x-coordinate of the object | -Inf | Inf |
| 4 | y-coordinate of the object | -Inf | Inf |
| 5 | z-coordinate of the object | -Inf | Inf |
| 6 | x-coordinate relative position between the gripper and the object | -Inf | Inf |
| 7 | y-coordinate relative position between the gripper and the object | -Inf | Inf |
| 8 | z-coordinate relative position between the gripper and the object | -Inf | Inf |
| 9 | Half of the distance between the fingers | -Inf | Inf |
| 10 | Half of the distance between the fingers | -Inf | Inf |
| 11 | x-coordinate of angle of the object | -Inf | Inf |
| 12 | y-coordinate of angle of the object | -Inf | Inf |
| 13 | z-coordinate of angle of the object | -Inf | Inf |
| 14 | x-coordinate velocity of the object | -Inf | Inf |
| 15 | y-coordinate velocity of the object | -Inf | Inf |
| 16 | z-coordinate velocity of the object | -Inf | Inf |
| 17 | x-coordinate angular velocity of the object | -Inf | Inf |
| 18 | y-coordinate angular velocity of the object | -Inf | Inf |
| 19 | z-coordinate angular velocity of the object | -Inf | Inf |
| 20 | x-coordinate velocity of the gripper | -Inf | Inf |
| 21 | y-coordinate velocity of the gripper | -Inf | Inf |
| 22 | z-coordinate velocity of the gripper | -Inf | Inf |
| 23 | Left finger relative motion to the gripper | -Inf | Inf |
| 24 | Right finger relative motion to the gripper | -Inf | Inf |

3, 4, 8 and 9 remains very small since the gripper is blocked closed.

### Achieved and desired goal

| Num | Observation | Control Min | Control Max |
|--|--|--|--|
| 0 | x-coordinate of the object | -Inf | Inf |
| 1 | y-coordinate of the object | -Inf | Inf |
| 2 | z-coordinate of the object | -Inf | Inf |


## Rewards

The reward is 0.0 when the object is within 5 cm of the goal position. It is -1.0 otherwise.
