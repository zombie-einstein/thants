<div align="center">
  <img src="https://github.com/zombie-einstein/thants/raw/main/.github/images/thants.gif" />
  <br>
  <em>Thanks ants!</em>
</div>
<br>

# Thants

*Multi-agent and multi-team reinforcement-learning environment modelling ant foraging*

## Introduction

Thants is a rl environment library based on models of ant colonies foraging for food, also supporting
environments with multiple competing colonies.

Thants is implemented using JAX, allowing the environments to be run on GPU enabling large scale performant
simulation, and the ability to run environments alongside JAX and Pytorch ML tools.

The environment is implemented using the [Jumanji](https://github.com/instadeepai/jumanji) RL environment API, with some modification
for the multi-colony case.

## Usage

### Installation

Thants can be installed from pypi using

```commandline
pip install thants
```

### Examples

#### Single Colony

The single colony environment follows the [Jumanji](https://github.com/instadeepai/jumanji)
environment API, with actions provided as an array of individual
actions:

```python
from thants.envs import ThantsMono
import jax

env = ThantsMono(dims=(50, 50))
key = jax.random.PRNGKey(101)
state, obs = env.reset(key)
state_history = [state]

for _ in range(50):
    key, action_key = jax.random.split(key, 2)
    actions = jax.random.choice(
        action_key, env.num_actions, (env.num_agents,)
    )
    state, obs = env.step(state, actions)
    state_history.append(state)

env.animate(state_history, 100, "mono_colony.gif")
```

#### Multi-Colony

In the multi-colony case each colony is treated independently (and can be
different sizes), so actions, observations, timesteps are list/tuples of
arrays/structs:

```python
from thants.envs import Thants
import jax
import jax.numpy as jnp

env = Thants((50, 100))
key = jax.random.PRNGKey(101)
state, obs = env.reset(key)
state_history = [state]

for _ in range(50):
    key, k1, k2 = jax.random.split(key, 3)
    # List of action arrays per colony
    actions = [
        jax.random.choice(k1, env.num_actions, (env.num_agents[0],)),
        jax.random.choice(k2, env.num_actions, (env.num_agents[1],)),
    ]
    state, obs = env.step(state, actions)
    state_history.append(state)

env.animate(state_history, 100, "multi_colony.gif")
```

Preset simple environments can be imported from `thants.envs.ThantsDual` and
`thants.envs.ThantsQuad` with 2 and 4 colonies respectively.

## Environment

<div align="center">
  <img src="https://github.com/zombie-einstein/thants/raw/main/.github/images/thants_env.gif" />
  <br>
  <em>A Thants environment with two competing colonies.</em>
</div>
<br>

The environment is modelled as a grid, wrapped at the boundaries. Ants (the agents)
occupy individual cells on the grid (and cannot overlap). Ants can pick up, carry,
and deposit food, or deposit persistent signals that can be observed by other ants
in the same colony.

### State

#### Colonies

The state of the ant colonies is represented by a single struct:

- *Ants*: Individual ants themselves have several components:
    - *Positions*: 2d indices of ant positions on the environment grid.
    - *Carrying*: The amount of food being carried by each ant.
    - *Health*: Ant health (currently unused).
- *Colony Index*: The index of the colony each ant belongs to
- *Nests*: 2d array indicating the index of the colony each cell belongs to
  (`0` in the case a cell is not the nest of any colony).
- *Signals*: 4d array of signal deposits at each cell for each colony. Signals have
  multiple channels to facilitate communication between ants (i.e. the 2nd dimension
  of the array is the signal channel).

#### Environment

The state of the environment then consists of the colonies and state shared
by all the colonies

- *Colonies*: Ant colonies state
- *Food*: 2d array representing the amount of food deposited at each cell
- *Terrain*: Array of flags indicating if a cell can be occupied by an ant. This
  allows obstacles to be placed on the environment.

#### Updates

Each step of the environment performs the following update to the state:

- Convert integer action choices into state updates
- Apply ant position updates
- Apply food pick-up and deposit actions
- Drop any new food deposits
- Dissipate and propagate signals
- Apply signal deposit actions
- Clear any food that has been deposited on a nest (i.e. the food is consumed
  by the colony)

The behaviour of the dynamics of signals can be customised by implementing the
[`thants.signals.SignalPropagator`](src/thants/signals.py) base class and
passing it when initialising the environment.

#### State Generators

The initialisation of the environment can be customised by implementing the
relevant base class and passing them to the environment.

### Actions

Ants can select from several discrete actions, indicated by an integer value:

- `0`: Null action (i.e. no change to the environment)
- `1 - 4`: Move in one of the four ordinal directions (if possible)
- `5`: Take a fixed amount of food from the ants location (if possible)
- `6`: Deposit a fixed amount of food from the ants location (if possible)
- `7+`: Deposit a fixed amount of signal at the ants location

Note that actions can be selected, but may not be possible e.g. attempting
to move to an occupied cell, or taking food from an empty cell. In this
case there will be no change in state due to the chosen action.

### Observations

Individual agent observations also consist of several components. Observations are
individually made for the local neighbourhood of each ant, i.e. the 8 surrounding
cells on the environment grid, and their own cell:

- `ants`: Flag indicating if a cell in the neighbourhood is occupied by an ant,
  with the shape `[n-ants, n-colonies, 9]` where the second dimensions indicates the
  individual colonies. The ants from the same colony will always be on the first row.
- `signals`: Signal deposits in the neighbourhood (across all channels), signals are
  observed individually for each colony.
- `food`: Food deposits within the neighbourhood.
- `nest`: Flag indicating if a neighbouring cell is designated as a nest for the ants colony.
- `terrain`: Flag indicating if a neighbouring cell can be occupied.
- `carrying`: Amount of food currently being carried by each ant.

### Rewards

By default, rewards are granted to ants when they deposit food on their colonies nest.
Reward signals can be customised by implementing the respective base classes
[`thants.rewards.RewardFn`](src/thants/rewards.py).
