<div align="center">
  <img src="https://github.com/zombie-einstein/thants/raw/main/.github/images/thants.gif" />
  <br>
  <em>Thanks ants!</em>
</div>
<br>

# Thants

*Multi-agent RL environment modelling ant foraging*

## Environment

The environment is modelled as a grid, wrapped at the boundaries. Ants (the agents)
occupy individual cells on the grid (and cannot overlap). Ants can pick up
and deposit food, and signals that can be observed by other ants.

The environment is implemented using the [Jumanji](https://github.com/instadeepai/jumanji) RL environment API.

### State

The state of the environment is represented by several component:

- *Ant states*: Individual ants have several components
    - *Positions*: 2d indices of ant positions on the environment grid
    - *Carrying*: The amount of food being carried by an ant
    - *Health*: Ant health (currently unused)
- *Food*: Array representing the amount of food deposited at each cell
- *Nest*: Array indicating if a cell is designated as a nest
- *Signals*: Array of signal deposits at each cell. Signals can have several channels
  to facilitate ant communication.

As well as updates due to agent actions, signal and food states are updated
according due to environmental factors, e.g. signal decay and dispersion, or
new food sources. These dynamics can be customised by implementing the relevant
base-clas/interface.

### Actions

Ants can select from several discrete actions, indicated by an integer value:

- `0`: Null action
- `1 - 4`: Move in one of the four ordinal directions if possible
- `5`: Take a fixed amount of food from the ants location (if possible)
- `6`: Deposit a fixed amount of food from the ants location (if possible)
- `7+`: Deposit a fixed amount of signal at the ants location

Note that actions can be selected, but may not be possible e.g. attempting
to move to an occupied cell, or taking food from an empty cell. In this
case there will be no change in state due to the chosen action.

### Observations

Individual agent observations also consist of several components. Observations are
made for the local neighbourhood of each ant, i.e. the 8 surrounding cells, and their
own cell:

- `ants`: Flag indicating if a cell in the neighbourhood is occupied by an ant
- `signals`: Signal deposits in the neighbourhood (across all channels)
- `food`: Food deposits within the neighbourhood
- `nest`: Flag indicating if a neighbouring cell is designated as a nest

### Rewards

Reward signals for individual agents can be customised by implementing the base class
`thants.rewards.RewardFn`.
