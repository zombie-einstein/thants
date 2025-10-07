import abc

import chex
import jax.numpy as jnp

from .types import State


class RewardFn(abc.ABC):
    @abc.abstractmethod
    def __call__(self, old_state: State, new_state: State) -> list[chex.Array]:
        """
        Generate observations for each colony

        Parameters
        ----------
        old_state
        new_state

        Returns
        -------
        list[chex.Array]
            List of rewards arrays per colony
        """


class NullRewardFn(RewardFn):
    def __call__(self, old_state: State, new_state: State) -> list[chex.Array]:
        return [jnp.zeros(c.ants.pos.shape[:1]) for c in new_state.colonies]
