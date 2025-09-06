import abc

import chex
import jax.numpy as jnp

from .types import State


class RewardFn(abc.ABC):
    @abc.abstractmethod
    def __call__(self, old_state: State, new_state: State) -> chex.Array:
        """Reward function should return rewards for each individual agent"""


class NullRewardFn(RewardFn):
    def __call__(self, old_state: State, new_state: State) -> chex.Array:
        return jnp.zeros_like(old_state.ants.health)
