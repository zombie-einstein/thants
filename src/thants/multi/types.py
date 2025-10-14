from typing import TYPE_CHECKING

import chex

from thants.common.types import Ants

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from chex import dataclass


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
