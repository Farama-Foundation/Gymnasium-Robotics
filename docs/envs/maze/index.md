---
firstpage:
lastpage:
---

# Maze

A collection of environments in which an agent has to navigate through a maze to reach certain goal position. Two different agents can be used: a 2-DoF force-controlled ball, or the classic `Ant` agent from the [Gymnasium MuJoCo environments](https://gymnasium.farama.org/environments/mujoco/ant/). The environment can be initialized with a variety of maze shapes with increasing levels of difficulty.

```{raw} html
    :file: list.html
```

## Reference

These environments were first introduced in [“D4RL: Datasets for Deep Data-Driven Reinforcement Learning”](https://arxiv.org/abs/2004.07219) by Justin Fu, Aviral Kumar, Ofir Nachum, George Tucker, Sergey Levine. Which can be cited as follows:

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
