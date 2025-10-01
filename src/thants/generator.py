import abc
import math

import chex
import jax
import jax.numpy as jnp
import jax.random

from .types import Ants
from .utils import get_rectangular_indices


class Generator(abc.ABC):
    """
    Base initial state generator and food updater
    """

    def __init__(self, dims: tuple[int, int], n_agents: int) -> None:
        """
        Initialise base attributes

        Parameters
        ----------
        dims
            Environment dimensions
        n_agents
            Number of ant agents to initialise
        """
        self.dims = dims
        self.n_agents = n_agents

    @abc.abstractmethod
    def init(self, key: chex.PRNGKey) -> tuple[Ants, chex.Array, chex.Array]:
        """
        Initialise ant, nest, and food states

        Parameters
        ----------
        key
            JAX random key

        Returns
        -------
        tuple[Ants, chex.Array, chex.Array]
            Tuple containing ant, nest, and food states respectively
        """

    @abc.abstractmethod
    def update_food(self, key: chex.PRNGKey, step: int, food: chex.Array) -> chex.Array:
        """
        Update food state during simulation, e.g. drop more food

        Parameters
        ----------
        key
            JAX random key
        step
            Simulation step
        food
            Current food state

        Returns
        -------
        chex.Array
            New food state
        """


class BasicGenerator(Generator):
    """
    Basic generator that creates rectangular nest and food blocks

    Generator that places:

    - A rectangular nest at the centre of the environment
    - A rectangular food block at a random location
    - Ants in a square region at the centre of the environment

    and then drops new blocks of food at random locations at fixed intervals.
    """

    def __init__(
        self,
        dims: tuple[int, int],
        n_agents: int,
        nest_dims: tuple[int, int],
        food_dims: tuple[int, int],
        drop_interval: int,
        drop_amount: float = 1.0,
    ) -> None:
        """
        Initialise a basic generator

        Parameters
        ----------
        dims
            Environment dimensions
        n_agents
            Number of ant agents to generate
        nest_dims
            Dimensions of the nest region
        food_dims
            Dimensions of food blocks
        drop_interval
            Interval at which new blocks are dropped
        drop_amount
            Amount of food in each cell of food blocks
        """
        assert n_agents <= dims[0] * dims[1]

        self.nest_dims = nest_dims
        self.food_dims = food_dims
        self.drop_interval = drop_interval
        self.drop_amount = drop_amount
        super().__init__(dims, n_agents)

    def _drop_food(self, key: chex.PRNGKey, food: chex.Array) -> chex.Array:
        """
        Place a new fixed size block of at a random location

        Parameters
        ----------
        key
            JAX random key
        food
            Current food state

        Returns
        -------
        chex.Array
            Updated food state
        """
        dims = jnp.array(self.dims)
        food_off = jax.random.randint(key, (2,), jnp.zeros((2,)), dims)
        food_idxs = get_rectangular_indices(self.food_dims)
        food_idxs = food_idxs + food_off
        food_idxs = food_idxs % dims
        food = food.at[food_idxs[:, 0], food_idxs[:, 1]].add(self.drop_amount)
        return food

    def init(self, key: chex.PRNGKey) -> tuple[Ants, chex.Array, chex.Array]:
        """
        Initialise ant, nest, and food states

        Initialises a new initial state with

        - A rectangular nest at the centre of the environment
        - Ants placed in an approximate square at the centre of the environment
        - A rectangular block of food at random location in the environment

        Parameters
        ----------
        key
            JAX random key

        Returns
        -------
        tuple[Ants, chex.Array, chex.Array]
            Tuple containing ant, nest, and food states respectively
        """
        dims = jnp.array(self.dims)
        centre = (dims // 2)[jnp.newaxis]

        d = math.ceil(math.sqrt(self.n_agents))
        ant_pos = get_rectangular_indices((d, d))[: self.n_agents]
        ant_pos = ant_pos + centre - jnp.array([[d, d]]) // 2
        ant_pos = ant_pos % dims

        ant_health = jnp.ones((self.n_agents,))
        ant_carrying = jnp.zeros((self.n_agents,))
        ants = Ants(pos=ant_pos, health=ant_health, carrying=ant_carrying)

        nest_idxs = get_rectangular_indices(self.nest_dims)
        nest_idxs = nest_idxs + centre - jnp.array(self.nest_dims) // 2
        nest_idxs = nest_idxs % dims
        nest = jnp.zeros(self.dims, dtype=bool)
        nest = nest.at[nest_idxs[:, 0], nest_idxs[:, 1]].set(True)

        food = jnp.zeros(self.dims, dtype=float)
        food = self._drop_food(key, food)

        return ants, nest, food

    def update_food(self, key: chex.PRNGKey, step: int, food: chex.Array) -> chex.Array:
        """
        Drop rectangular blocks of food at random locations at fixed intervals

        Parameters
        ----------
        key
            JAX random key
        step
            Simulation step
        food
            Current food state

        Returns
        -------
        chex.Array
            New food state
        """
        return jax.lax.cond(
            (step + 1) % self.drop_interval == 0,
            self._drop_food,
            lambda k, f: f,
            key,
            food,
        )
