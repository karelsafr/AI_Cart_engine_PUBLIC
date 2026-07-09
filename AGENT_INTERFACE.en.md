# AGENT_INTERFACE.md

This document describes the stable interface between the AI Cart V2 engine and a student agent.

It is a contract: if an agent follows the rules below, the engine can automatically discover it, run it in training, save it, load it, use it in a duel, and use it in a benchmark.

## Location and name

A student agent belongs in:

```text
students/agents/
```

The file and the class must have the same name:

```text
students/agents/AIbrain_MyTeam.py
class AIbrain_MyTeam
```

The agent name is entered in the menu, benchmark YAML, and duel without `.py`:

```text
AIbrain_MyTeam
```

The engine looks for agents in this order:

1. `students/agents/`
2. `AI_engines/`

If the same name exists in both folders, the student version is used.

## Constructor

The agent must be creatable without arguments:

```python
brain = AIbrain_MyTeam()
```

In `__init__`, the agent should prepare everything it needs for decisions, mutations, scoring, and saving.

Recommended minimum:

```python
self.NAME = "MyTeam"
self.score = 0.0
self.store()
```

## Required attributes

The engine expects:

- `NAME` - human-readable brain name for output, benchmark, and debug
- `score` - current fitness value of the agent

`score` should be a number. Higher means better.

## Required methods

The registry checks that the agent has these methods:

```python
def decide(self, data): ...
def mutate(self): ...
def store(self): ...
def get_parameters(self): ...
def set_parameters(self, parameters): ...
def calculate_score(self, distance, time, no): ...
def passcardata(self, x, y, speed): ...
def getscore(self): ...
```

The typed version of the same interface is in `core/protocols.py` as `AgentProtocol`.

## decide(data)

Called when controlling the car.

```python
def decide(self, data):
    return [up, down, left, right]
```

`data` is a sequence of raycast distances in tiles. The engine currently uses 9 rays:

```text
[-90, -45, -20, -5, 0, 5, 20, 45, 90]
```

The output must have at least 4 numeric values:

```text
index 0 - throttle
index 1 - brake
index 2 - left
index 3 - right
```

The car performs an action if the corresponding value is greater than `0.5`.

Example:

```python
return [1.0, 0.0, 0.0, 0.8]
```

This output means throttle and steering right.

## passcardata(x, y, speed)

Called before `decide`.

```python
def passcardata(self, x, y, speed):
    self.x = x
    self.y = y
    self.speed = speed
```

Parameters:

- `x` - current x position of the car in pixels
- `y` - current y position of the car in pixels
- `speed` - current car speed

The agent may use these values for decisions or scoring. If it does not need them, it can simply store or ignore them.

## calculate_score(distance, time, no)

Called during driving after the car moves.

```python
def calculate_score(self, distance, time, no):
    self.score = distance
```

Parameters:

- `distance` - total distance traveled by the car in tiles
- `time` - car runtime in seconds
- `no` - helper engine value; students usually do not need it

This method should update `self.score`. Training then selects better agents based on the score.

## getscore()

Returns the current score:

```python
def getscore(self):
    return self.score
```

The engine uses the score when ranking cars in training.

## mutate()

Called when creating a new generation.

```python
def mutate(self):
    ...
    self.store()
```

The method should randomly change the agent parameters. After changing parameters, it is recommended to call `self.store()` so the current state can be saved into `.npz`.

Mutation must not change the agent interface. After mutation, the agent must still support `decide`, `get_parameters`, and `set_parameters`.

## store()

Prepares parameters for saving:

```python
def store(self):
    self.parameters = {
        "W": self.W,
        "b": self.b,
        "NAME": self.NAME,
    }
```

Store only values that `numpy.savez` can save into `.npz`: numbers, strings, numpy arrays, or simple combinations of these values.

## get_parameters()

Returns parameters for `SAVE`:

```python
def get_parameters(self):
    return self.parameters
```

During saving, training calls:

```python
np.savez(save_path, **brain.get_parameters())
```

Therefore `get_parameters()` must return a dictionary.

## set_parameters(parameters)

Restores an agent from a save:

```python
def set_parameters(self, parameters):
    self.W = parameters["W"]
    self.b = parameters["b"]
    self.NAME = str(parameters["NAME"])
    self.store()
```

`parameters` can be:

- `numpy.lib.npyio.NpzFile` from `np.load(...)`
- a parameter dictionary from another agent

The method must set the internal state so the next `decide` call works. If the save does not match the agent architecture, the method may raise an error; the engine catches it and prints a message.

## Save file

Saves belong in:

```text
students/saves/
```

Recommended name:

```text
AIbrain_MyTeam.npz
```

A save must match the agent architecture. Parameters saved from one type of agent may not load into another type of agent.

## Minimal agent skeleton

```python
from __future__ import annotations

import copy
import numpy as np


class AIbrain_MyTeam:
    def __init__(self) -> None:
        self.NAME = "MyTeam"
        self.score = 0.0
        self.W = np.zeros((4, 9))
        self.b = np.zeros(4)
        self.store()

    def decide(self, data):
        x = np.asarray(data, dtype=float)
        y = self.W.dot(x) + self.b
        return y

    def mutate(self):
        self.W += np.random.normal(0, 0.05, self.W.shape)
        self.b += np.random.normal(0, 0.05, self.b.shape)
        self.store()

    def store(self):
        self.parameters = {
            "W": self.W,
            "b": self.b,
            "NAME": self.NAME,
        }

    def get_parameters(self):
        return copy.deepcopy(self.parameters)

    def set_parameters(self, parameters):
        self.W = np.array(parameters["W"], dtype=float)
        self.b = np.array(parameters["b"], dtype=float)
        self.NAME = str(parameters["NAME"])
        self.store()

    def calculate_score(self, distance, time, no):
        self.score = distance

    def passcardata(self, x, y, speed):
        self.x = x
        self.y = y
        self.speed = speed

    def getscore(self):
        return self.score
```

## What the engine guarantees

The engine guarantees:

- the agent is created without arguments
- during driving, it calls `passcardata`, then `decide`, then `calculate_score`
- during saving, it calls `get_parameters`
- during loading, it calls `set_parameters`
- when creating a new generation, it uses a copy of parameters and `mutate`
- agents from `students/agents/` have priority over built-in agents

## What not to rely on

A student should not rely on internal engine details:

- the exact implementation of `Car_manager`
- the exact ordering of cars in sprite groups
- absolute paths outside `students/`
- direct imports of internal scenes
- `no` in `calculate_score` keeping the same meaning forever

The stable part is the interface described in this document.

## Reference

The best current reference is:

```text
AI_engines/AIbrain_linear.py
```

For deeper interface checks, use:

```text
core/protocols.py
core/AgentRegistry.py
```
