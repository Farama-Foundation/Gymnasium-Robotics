from gym.envs.registration import register

from .kitchen_env import (
    KitchenMicrowaveKettleBottomBurnerLightV0,
    KitchenMicrowaveKettleLightSliderV0,
)

# Smaller dataset with only positive demonstrations.
register(
    id="kitchen-complete-v0",
    entry_point="d4rl.kitchen:KitchenMicrowaveKettleLightSliderV0",
    max_episode_steps=280,
    kwargs={
        "ref_min_score": 0.0,
        "ref_max_score": 4.0,
        "dataset_url": "http://rail.eecs.berkeley.edu/datasets/offline_rl/kitchen/mini_kitchen_microwave_kettle_light_slider-v0.hdf5",
    },
)

# Whole dataset with undirected demonstrations. A subset of the demonstrations
# solve the task.
register(
    id="kitchen-partial-v0",
    entry_point="d4rl.kitchen:KitchenMicrowaveKettleLightSliderV0",
    max_episode_steps=280,
    kwargs={
        "ref_min_score": 0.0,
        "ref_max_score": 4.0,
        "dataset_url": "http://rail.eecs.berkeley.edu/datasets/offline_rl/kitchen/kitchen_microwave_kettle_light_slider-v0.hdf5",
    },
)

# Whole dataset with undirected demonstrations. No demonstration completely
# solves the task, but each demonstration partially solves different
# components of the task.
register(
    id="kitchen-mixed-v0",
    entry_point="d4rl.kitchen:KitchenMicrowaveKettleBottomBurnerLightV0",
    max_episode_steps=280,
    kwargs={
        "ref_min_score": 0.0,
        "ref_max_score": 4.0,
        "dataset_url": "http://rail.eecs.berkeley.edu/datasets/offline_rl/kitchen/kitchen_microwave_kettle_bottomburner_light-v0.hdf5",
    },
)
