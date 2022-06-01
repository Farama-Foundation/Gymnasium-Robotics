# Spaces

```{eval-rst}
.. autoclass:: gym.spaces.Space
```

## General Functions

Each space implements the following functions:

```{eval-rst}
.. autofunction:: gym.spaces.Space.sample

.. autofunction:: gym.spaces.Space.contains

.. autoproperty:: gym.spaces.Space.shape

.. property:: gym.spaces.Space.dtype

    Return the data type of this space.

.. autofunction:: gym.spaces.Space.seed

.. autofunction:: gym.spaces.Space.to_jsonable

.. autofunction:: gym.spaces.Space.from_jsonable
``` 

## Box

```{eval-rst}
.. autoclass:: gym.spaces.Box
    
    .. automethod:: __init__
    .. automethod:: is_bounded
    .. automethod:: sample
``` 

## Discrete

```{eval-rst}
.. autoclass:: gym.spaces.Discrete
 
    .. autoclass:: __init__
``` 

## MultiBinary

```{eval-rst}
.. autoclass:: gym.spaces.MultiBinary
``` 

## MultiDiscrete

```{eval-rst}
.. autoclass:: gym.spaces.MultiDiscrete

    .. automethod:: __init__
``` 

## Dict

```{eval-rst}
.. autoclass:: gym.spaces.Dict

    .. automethod:: __init__
``` 

## Tuple

```{eval-rst}
.. autoclass:: gym.spaces.Tuple

    .. automethod:: __init__
``` 

## Utility Functions

```{eval-rst}
.. autofunction:: gym.spaces.utils.flatdim

.. autofunction:: gym.spaces.utils.flatten_space

.. autofunction:: gym.spaces.utils.flatten

.. autofunction:: gym.spaces.utils.unflatten
``` 