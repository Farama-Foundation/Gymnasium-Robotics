---
firstpage:
lastpage:
---

# Franka Kitchen

Multitask environment in which a 9-DoF Franka robot is placed in a kitchen containing several common household items. The goal of each task is to interact with the items in order to reach a desired goal configuration.

```{raw} html
    :file: list.html
```

The tasks can be selected when the environment is initialized with the `tasks_to_complete` list argument as follows:

```python

import gymnasium as gym

env = gym.make('FrankaKitchen-v1', tasks_to_complete=['microwave', 'kettle', 'bottom_left_burner'])
```

The possible tasks to complete are:

* `bottom_right_burner`
* `bottom_left_burner`
* `top_right_burner`
* `top_left_burner`
* `light_switch`
* `slide_cabinet`
* `left_hinge_cabinet`
* `right_hinge_cabinet`
* `microwave`
* `kettle`

## References

These environments were first introduced in [“Relay Policy Learning: Solving Long-Horizon Tasks via Imitation and Reinforcement Learning”](https://arxiv.org/abs/1910.11956) by Abhishek Gupta, Vikash Kumar, Corey Lynch, Sergey Levine, Karol Hausman, and later modified in [“D4RL: Datasets for Deep Data-Driven Reinforcement Learning”](https://arxiv.org/abs/2004.07219) by Justin Fu, Aviral Kumar, Ofir Nachum, George Tucker, Sergey Levine. Both publications can be cited as follows:

```
@article{gupta2019relay,
  title={Relay policy learning: Solving long-horizon tasks via imitation and reinforcement learning},
  author={Gupta, Abhishek and Kumar, Vikash and Lynch, Corey and Levine, Sergey and Hausman, Karol},
  journal={arXiv preprint arXiv:1910.11956},
  year={2019}
}
```

```
@misc{fu2020d4rl,
    title={D4RL: Datasets for Deep Data-Driven Reinforcement Learning},
    author={Justin Fu and Aviral Kumar and Ofir Nachum and George Tucker and Sergey Levine},
    year={2020},
    eprint={2004.07219},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```

```{toctree}
:glob:
:hidden:
./*