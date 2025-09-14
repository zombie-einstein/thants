import abc
import math

import chex
import jax
import jax.numpy as jnp
import jax.random

from .types import Ants


class Generator(abc.ABC):
    def __init__(self, dims: tuple[int, int], n_agents: int) -> None:
        self.dims = dims
        self.n_agents = n_agents

    @abc.abstractmethod
    def init(self, key: chex.PRNGKey) -> tuple[Ants, chex.Array, chex.Array]:
        """Generate initial state"""

    @abc.abstractmethod
    def update_food(self, key: chex.PRNGKey, step: int, food: chex.Array) -> chex.Array:
        """Update food"""


class BasicGenerator(Generator):
    def __init__(
        self,
        dims: tuple[int, int],
        n_agents: int,
        nest_dims: tuple[int, int],
        food_dims: tuple[int, int],
        drop_interval: int,
        drop_amount: float = 1.0,
    ) -> None:
        assert n_agents <= dims[0] * dims[1]

        self.nest_dims = nest_dims
        self.food_dims = food_dims
        self.drop_interval = drop_interval
        self.drop_amount = drop_amount
        super().__init__(dims, n_agents)

    def _drop_food(self, key: chex.PRNGKey, food: chex.Array) -> chex.Array:
        n_food = math.prod(self.food_dims)
        dims = jnp.array(self.dims)
        food_off = jax.random.randint(key, (2,), jnp.zeros((2,)), dims)
        food_idxs = jnp.indices(self.food_dims).reshape(2, n_food).T
        food_idxs = food_idxs + food_off
        food_idxs = food_idxs % dims
        food = food.at[food_idxs[:, 0], food_idxs[:, 1]].add(self.drop_amount)
        return food

    def init(self, key: chex.PRNGKey) -> tuple[Ants, chex.Array, chex.Array]:
        """Generate initial state"""

        dims = jnp.array(self.dims)
        centre = (dims // 2)[jnp.newaxis]

        d = math.isqrt(self.n_agents) + 1
        ant_pos = jnp.indices((d, d)).reshape(2, d * d).T[: self.n_agents]
        ant_pos = ant_pos + centre - jnp.array([[d, d]]) // 2
        ant_pos = ant_pos % dims

        ant_health = jnp.ones((self.n_agents,))
        ant_carrying = jnp.zeros((self.n_agents,))
        ants = Ants(pos=ant_pos, health=ant_health, carrying=ant_carrying)

        n_nest = math.prod(self.nest_dims)
        nest_idxs = jnp.indices(self.nest_dims).reshape(2, n_nest).T
        nest_idxs = nest_idxs + centre - jnp.array(self.nest_dims) // 2
        nest_idxs = nest_idxs % dims
        nest = jnp.zeros(self.dims, dtype=bool)
        nest = nest.at[nest_idxs[:, 0], nest_idxs[:, 1]].set(True)

        food = jnp.zeros(self.dims, dtype=float)
        food = self._drop_food(key, food)

        return ants, nest, food

    def update_food(self, key: chex.PRNGKey, step: int, food: chex.Array) -> chex.Array:
        return jax.lax.cond(
            (step + 1) % self.drop_interval == 0,
            self._drop_food,
            lambda k, f: f,
            key,
            food,
        )
