from typing import TYPE_CHECKING

import chex

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from chex import dataclass


@dataclass
class Ants:
    """
    Ant states

    pos: Indices of ant grid positions
    health: Ant health values
    carrying: Amount of food being carried by ants
    """

    pos: chex.Array  # (n_ants, 2)
    health: chex.Array  # (n_ants,)
    carrying: chex.Array  # (n_ants,)


@dataclass
class Colony:
    ants: Ants
    signals: chex.Array  # [n-channels, *env-size]
    nest: chex.Array  # [*env-size]


@dataclass
class Colonies:
    """
    Joint state of multiple colonies

    ants: State of all ants
    colony_idx: Idx assigning each ant to a colony
    signals: Signals states for each colony
    nests: Nest of indices indicating if a cell is a nest of a colony
    """

    ants: Ants
    colony_idx: chex.Array  # [n-ants,]
    signals: chex.Array  # [n-colonies, n-channels, *env-size]
    nests: chex.Array  # [*env-size]


@dataclass
class State:
    """
    Environment state

    step: Environment step
    key: JAX random key
    colonies: List ant colonies
    food: Environment food deposit state
    terrain: Flag indicating if a cell is passable by ants
    """

    step: int
    key: chex.PRNGKey
    colonies: Colonies
    food: chex.Array  # [*env-size]
    terrain: chex.Array  # [*env-size]


@dataclass
class SignalActions:
    """
    Agent deposit signal actions

    channel: Channel to deposit signals on
    amount: Amount to deposit
    """

    channel: chex.Array  # (n_ants,)
    amount: chex.Array  # (n_ants,)


@dataclass
class Actions:
    """
    Ant action environment updates

    movements: Movement actions to apply to ant positions
    take_food: Food amounts to be picked up by ants
    deposit_food: Food amounts to be dropped by ants
    deposit_signals: Signal deposit channels and amounts
    """

    movements: chex.Array  # (n_ants, 2)
    take_food: chex.Array  # (n_ants,)
    deposit_food: chex.Array  # (n_ants,)
    deposit_signals: SignalActions


@dataclass
class Observations:
    """
    Agent observations

    ants: Flag indicating if cells in the local neighbourhood are occupied by an ant
    food: Amount of food deposited in neighbouring cells
    signals: Signal deposits in neighbouring cells
    nest: Flag indicating if cells in the local neighbourhood are nest cells
    terrain: Passable/impassable cells in the neighbourhood
    carrying: Food amount being held
    """

    ants: chex.Array  # (n_ants, 9)
    food: chex.Array  # (n_ants, 9)
    signals: chex.Array  # (n_ants, n-channels, 9)
    nest: chex.Array  # (n_ants, 9)
    terrain: chex.Array  # (n_ants, 9)
    carrying: chex.Array  # (n_ants,)


@dataclass
class ColorScheme:
    """
    Visualisation color-scheme, given as an array of rgb values

    ants: Array of colors for each colony
    food: Color to represent food
    terrain: Array of blocked/passable terrain cells
    """

    ants: chex.Array  # (n-colonies, 4)
    food: chex.Array  # (4,)
    terrain: chex.Array  # (2, 4)
