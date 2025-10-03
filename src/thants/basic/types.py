from typing import TYPE_CHECKING

import chex

from thants.common.types import Ants

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
