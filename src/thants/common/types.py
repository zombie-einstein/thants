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
class SignalActions:
    """
    Agent deposit signal actions

    channel: Channel to deposit signals on
    amount: Amount to deposit
    """

    channel: chex.Array  # (n_ants,)
    amount: chex.Array  # (n_ants,)
