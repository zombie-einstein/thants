import math
from dataclasses import dataclass
from typing import Sequence

import chex
import jax.numpy as jnp

from thants.types import Ants, Colony


def get_rectangular_indices(rec_dims: tuple[int, int]) -> chex.Array:
    """
    Get cell indices for a rectangle

    Parameters
    ----------
    rec_dims
        Dimensions of the rectangle

    Returns
    -------
    chex.Array
        Array of indices with shape [n, 2]
    """
    n_idxs = rec_dims[0] * rec_dims[1]
    idxs = jnp.indices(rec_dims).reshape(2, n_idxs).T
    return idxs


@dataclass
class BBox:
    """Rectangular region bounding box"""

    x0: tuple[int, int]
    x1: tuple[int, int]


def init_colony(
    dims: tuple[int, int],
    bounds: BBox,
    nest_dims: tuple[int, int],
    n_agents: int,
    n_signals: int,
) -> Colony:
    """
    Initialise a colony at the centre of a rectangular region

    Parameters
    ----------
    dims
        Environment dimensions
    bounds
        Bounding box of the rectangular region
    nest_dims
        Rectangular nest dimensions
    n_agents
        Number of agents
    n_signals
        Number of signal channels

    Returns
    -------
    Colony
        Initialised colony
    """
    x0 = jnp.array(bounds.x0)
    x1 = jnp.array(bounds.x1)
    dims = jnp.array(dims)
    centre = (x0 + ((x1 - x0) // 2))[jnp.newaxis]
    d = math.ceil(math.sqrt(n_agents))
    ant_pos = get_rectangular_indices((d, d))[:n_agents]
    ant_pos = ant_pos + centre - (jnp.array([[d, d]]) // 2)
    ant_pos = ant_pos % dims

    ant_health = jnp.ones((n_agents,))
    ant_carrying = jnp.zeros((n_agents,))

    ants = Ants(pos=ant_pos, health=ant_health, carrying=ant_carrying)

    nest_idxs = get_rectangular_indices(nest_dims)
    nest_idxs = nest_idxs + centre - jnp.array(nest_dims) // 2
    nest_idxs = nest_idxs % dims
    nest = jnp.zeros(dims, dtype=bool)
    nest = nest.at[nest_idxs[:, 0], nest_idxs[:, 1]].set(True)

    signals = jnp.zeros((n_signals, *dims))

    return Colony(ants=ants, signals=signals, nest=nest)


def init_colonies(
    env_dims: tuple[int, int],
    nest_dims: tuple[int, int],
    n_agents: Sequence[BBox],
    n_signals: int,
    bounds: Sequence[BBox],
) -> Sequence[Colony]:
    """
    Initialise multiple rectangular colonies

    Parameters
    ----------
    env_dims
        Environment dimensions
    nest_dims
        Rectangular nest dimensions
    n_agents
        Number of agents in each colony
    n_signals
        Number of signal channels
    bounds
        Bounding boxes of each colony

    Returns
    -------
    Sequence[Colony]
        List of initialised colonies
    """
    return [
        init_colony(
            env_dims,
            b,
            nest_dims,
            n,
            n_signals,
        )
        for b, n in zip(bounds, n_agents)
    ]
