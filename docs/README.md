# Gymnasium-Robotics docs


This repo contains the documentation for [Gymnasium-Robotics](https://github.com/Farama-Foundation/Gymnasium-Robotics).


To modify an environment follow the steps below. For more information about how to contribute to the documentation go to our [CONTRIBUTING.md](https://github.com/Farama-Foundation/Celshast/blob/main/CONTRIBUTING.md)

## Instructions for modifying pages

### Editing a page

Fork Gymnasium-Robotics and edit the docstring in the environment's Python file. Then, pip install your fork and run `docs/_scripts/gen_mds.py` in this repo. This will automatically generate a md documentation file for the environment.

## Build the Documentation

Install the required packages and Gym (or your fork):

```
pip install -r requirements.txt
pip install gym
```

To build the documentation once:

```
cd docs
make dirhtml
```

To rebuild the documentation automatically every time a change is made:

```
cd docs
sphinx-autobuild -b dirhtml . _build
```
