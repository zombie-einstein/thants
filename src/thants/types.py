from typing import TYPE_CHECKING

import chex

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from chex import dataclass


@dataclass
class Ants:
    pos: chex.Array  # (n_ants, 2)
    health: chex.Array  # (n_ants,)
    carrying: chex.Array  # (n_ants,)


@dataclass
class State:
    ants: Ants
    food: chex.Array
    signals: chex.Array
    nest: chex.Array


@dataclass
class Observation:
    local: chex.Array  # (n_ants, 9 * 3)
