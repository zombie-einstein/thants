import abc

import chex
import jax.numpy as jnp

from .types import Ants


class Generator(abc.ABC):
    def __init__(self, dims: tuple[int, int], n_agents: int) -> None:
        self.dims = dims
        self.n_agents = n_agents

    @abc.abstractmethod
    def __call__(self, key: chex.PRNGKey) -> tuple[Ants, chex.Array, chex.Array]:
        """Generate initial state"""


class BasicGenerator(Generator):
    def __call__(self, key: chex.PRNGKey) -> tuple[Ants, chex.Array, chex.Array]:
        """Generate initial state"""
        ant_pos = jnp.indices((5, self.n_agents // 5)).reshape(self.n_agents, 2)
        ant_health = jnp.ones((self.n_agents,))
        ant_carrying = jnp.zeros((self.n_agents,))
        ants = Ants(pos=ant_pos, health=ant_health, carrying=ant_carrying)
        nest = jnp.zeros(self.dims, dtype=bool)
        food = jnp.zeros(self.dims, dtype=float)

        return ants, nest, food
