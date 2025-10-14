from typing import Optional

import chex
import jax.numpy as jnp


def delivered_food(
    nest: chex.Array,
    pos: chex.Array,
    carrying_before: chex.Array,
    carrying_after: chex.Array,
    colony_idxs: Optional[chex.Array] = None,
) -> chex.Array:
    """
    Calculate food deposited by individual ant on a nest

    Parameters
    ----------
    nest
        Array of nest flags
    pos
        Array of ant positions
    carrying_before
        Food carried by ants at the start of the step
    carrying_after
        Food carried at the end of the step
    colony_idxs
        Colony indices of individual ants

    Returns
    -------
    chex.Array
        Array represented deposited food by each agent
    """
    d_carrying = carrying_before - carrying_after

    if colony_idxs is None:
        is_nest = nest.at[pos[:, 0], pos[:, 1]].get()
    else:
        is_nest = nest.at[pos[:, 0], pos[:, 1]].get()
        is_nest = (is_nest - 1) == colony_idxs

    rewards = jnp.where(is_nest, d_carrying, 0.0)
    return rewards
