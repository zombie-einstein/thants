import abc

import chex
import jax.numpy as jnp

from thants.common.rewards import delivered_food
from thants.multi.types import State


class RewardFn(abc.ABC):
    @abc.abstractmethod
    def __call__(self, old_state: State, new_state: State) -> list[chex.Array]:
        """
        Generate individual ant rewards for each colony

        Parameters
        ----------
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
    def __call__(self, old_state: State, new_state: State) -> list[chex.Array]:
        """
        Assigns 0 reward to all agents

        Parameters
        ----------
        old_state
            State at the start of the step
        new_state
            State at the end of the step

        Returns
        -------
        list[chex.Array]
            List of rewards arrays per colony
        """
        return [jnp.zeros(c.ants.pos.shape[:1]) for c in new_state.colonies]


class DeliveredFoodRewards(RewardFn):
    def __call__(self, old_state: State, new_state: State) -> list[chex.Array]:
        """
        Assigns rewards for ants depositing food on their own nest

        Parameters
        ----------
        old_state
            State at the start of the step
        new_state
            State at the end of the step

        Returns
        -------
        list[chex.Array]
            List of rewards arrays per colony
        """
        return [
            delivered_food(
                new_colony.nest,
                new_colony.ants.pos,
                old_colony.ants.carrying,
                new_colony.ants.carrying,
            )
            for old_colony, new_colony in zip(old_state.colonies, new_state.colonies)
        ]
