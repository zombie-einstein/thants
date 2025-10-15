import abc
from typing import Optional, Sequence

import chex
import jax.numpy as jnp
import numpy as np

from thants.types import State


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


def _get_boundaries(colony_sizes: Sequence[int]) -> np.typing.NDArray:
    boundaries = [0, *colony_sizes]
    boundaries = np.array(boundaries)
    boundaries = np.cumsum(boundaries)
    return boundaries


class RewardFn(abc.ABC):
    @abc.abstractmethod
    def __call__(
        self, colony_sizes: Sequence[int], old_state: State, new_state: State
    ) -> Sequence[chex.Array]:
        """
        Generate individual ant rewards for each colony

        Parameters
        ----------
        colony_sizes
            List of colony sizes (i.e. number of ants in each colony)
        old_state
            State at the start of the step
        new_state
            State at the end of the step

        Returns
        -------
        list[chex.Array]
            List of rewards arrays per colony
        """


class NullRewardFn(RewardFn):
    def __call__(
        self, colony_sizes: Sequence[int], old_state: State, new_state: State
    ) -> Sequence[chex.Array]:
        """
        Assigns 0 reward to all agents

        Parameters
        ----------
        colony_sizes
            List of colony sizes (i.e. number of ants in each colony)
        old_state
            State at the start of the step
        new_state
            State at the end of the step

        Returns
        -------
        Sequence[chex.Array]
            List of rewards arrays per colony
        """
        return [jnp.zeros(n) for n in colony_sizes]


class DeliveredFoodRewards(RewardFn):
    def __call__(
        self, colony_sizes: Sequence[int], old_state: State, new_state: State
    ) -> Sequence[chex.Array]:
        """
        Assigns rewards for ants depositing food on their own nest

        Parameters
        ----------
        colony_sizes
            List of colony sizes (i.e. number of ants in each colony)
        old_state
            State at the start of the step
        new_state
            State at the end of the step

        Returns
        -------
        Sequence[chex.Array]
            List of rewards arrays per colony
        """
        boundaries = _get_boundaries(colony_sizes)
        rewards = delivered_food(
            new_state.colonies.nests,
            new_state.colonies.ants.pos,
            old_state.colonies.ants.carrying,
            new_state.colonies.ants.carrying,
            colony_idxs=new_state.colonies.colony_idx,
        )
        return [rewards[a:b] for a, b in zip(boundaries[:-1], boundaries[1:])]
