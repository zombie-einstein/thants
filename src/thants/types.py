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
    step: int
    ants: Ants
    food: chex.Array
    signals: chex.Array
    nest: chex.Array


@dataclass
class Actions:
    movements: chex.Array  # (n_ants, 2)
    take_food: chex.Array  # (n_ants,)
    deposit_food: chex.Array  # (n_ants,)
    deposit_signals: chex.Array  # (n_ants,)


@dataclass
class Observations:
    ants: chex.Array  # (n_ants, 9)
    food: chex.Array  # (n_ants, 9)
    signals: chex.Array  # (n_ants, 9)
    nest: chex.Array  # (n_ants, 9)
    carrying: chex.Array  # (n_ants,)
