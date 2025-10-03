from typing import TYPE_CHECKING

import chex

from thants.common.types import Ants, SignalActions

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from chex import dataclass


@dataclass
class State:
    """
    Environment state

    step: Environment step
    key: JAX random key
    ants: Ant states
    food: Environment food deposit state
    signals: Ant deposited signals
    nest: FLag indicating nest designated cells
    terrain: Flag indicating if a cell is passable by ants
    """

    step: int
    key: chex.PRNGKey
    ants: Ants
    food: chex.Array  # [*env-size]
    signals: chex.Array  # [n-signal-channels, *env-size]
    nest: chex.Array  # [*env-size]
    terrain: chex.Array  # [*env-size]


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
