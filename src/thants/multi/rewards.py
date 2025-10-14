import abc

import chex
import jax.numpy as jnp
import numpy as np

from thants.common.rewards import delivered_food
from thants.multi.types import State


def _get_boundaries(colony_sizes: list[int]) -> np.typing.NDArray:
    boundaries = [0] + colony_sizes
    boundaries = np.array(boundaries)
    boundaries = np.cumsum(boundaries)
    return boundaries


class RewardFn(abc.ABC):
    @abc.abstractmethod
    def __call__(
        self, colony_sizes: list[int], old_state: State, new_state: State
    ) -> list[chex.Array]:
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
        self, colony_sizes: list[int], old_state: State, new_state: State
    ) -> list[chex.Array]:
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
        list[chex.Array]
            List of rewards arrays per colony
        """
        return [jnp.zeros(n) for n in colony_sizes]


class DeliveredFoodRewards(RewardFn):
    def __call__(
        self, colony_sizes: list[int], old_state: State, new_state: State
    ) -> list[chex.Array]:
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
        list[chex.Array]
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
