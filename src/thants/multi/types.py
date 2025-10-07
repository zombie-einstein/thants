from typing import TYPE_CHECKING

import chex

from thants.common.types import Colony

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
    colonies: List ant colonies
    food: Environment food deposit state
    terrain: Flag indicating if a cell is passable by ants
    """

    step: int
    key: chex.PRNGKey
    colonies: list[Colony]
    food: chex.Array  # [*env-size]
    terrain: chex.Array  # [*env-size]
