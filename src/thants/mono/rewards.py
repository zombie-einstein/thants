import abc

import chex
import jax.numpy as jnp

from thants.common.rewards import delivered_food
from thants.mono.types import State


class RewardFn(abc.ABC):
    """
    Base reward function
    """

    @abc.abstractmethod
    def __call__(self, old_state: State, new_state: State) -> chex.Array:
        """
        Generates rewards from old and new states

        Parameters
        ----------
        old_state
            State at the start of the step
        new_state
            State at the end of the step

        Returns
        -------
        chex.Array
            Array of individual agent rewards
        """


class NullRewardFn(RewardFn):
    """
    Dummy reward function returning 0 rewards
    """

    def __call__(self, old_state: State, new_state: State) -> chex.Array:
        """
        Generates fixed 0 rewards for all agents

        Parameters
        ----------
        old_state
            State at the start of the step
        new_state
            State at the end of the step

        Returns
        -------
        chex.Array
            Array of individual agent rewards
        """
        return jnp.zeros_like(old_state.colony.ants.health)


class DeliveredFoodRewards(RewardFn):
    def __call__(self, old_state: State, new_state: State) -> chex.Array:
        """
        Provide rewards when ants deposit food on the nest

        Parameters
        ----------
        old_state
            State at the start of the step
        new_state
            State at the end of the step

        Returns
        -------
        chex.Array
            Array of individual agent rewards
        """
        return delivered_food(
            new_state.colony.nest,
            new_state.colony.ants.pos,
            old_state.colony.ants.carrying,
            new_state.colony.ants.carrying,
        )
