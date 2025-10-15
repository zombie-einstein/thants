import abc

import chex
import jax
import jax.numpy as jnp

from thants.generators.colonies.utils import get_rectangular_indices


class FoodGenerator(abc.ABC):
    """
    base food state generator and updater
    """

    @abc.abstractmethod
    def init(self, dims: tuple[int, int], key: chex.PRNGKey) -> chex.Array:
        """
        Initialise environment food state

        Parameters
        ----------
        dims
            Environment dimensions
        key
            JAX random key

        Returns
        -------
        chex.Array
            Food state array
        """

    @abc.abstractmethod
    def update(self, key: chex.PRNGKey, step: int, food: chex.Array) -> chex.Array:
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
            Updated food state
        """


class BasicFoodGenerator(FoodGenerator):
    """
    Basic food generator

    Generator that initialises rectangular food blocks at fixed intervals.
    """

    def __init__(
        self,
        food_dims: tuple[int, int],
        drop_interval: int,
        drop_amount: float = 1.0,
    ):
        """
        Initialise basic food generator

        Parameters
        ----------
        food_dims
            Dimensions of food blocks
        drop_interval
            Interval at which new blocks are dropped
        drop_amount
            Amount of food in each cell of food blocks
        """
        self.food_dims = food_dims
        self.drop_interval = drop_interval
        self.drop_amount = drop_amount

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
        dims = jnp.array(food.shape)
        food_off = jax.random.randint(key, (2,), jnp.zeros((2,)), dims)
        food_idxs = get_rectangular_indices(self.food_dims)
        food_idxs = food_idxs + food_off
        food_idxs = food_idxs % dims
        food = food.at[food_idxs[:, 0], food_idxs[:, 1]].add(self.drop_amount)
        return food

    def init(self, dims: tuple[int, int], key: chex.PRNGKey) -> chex.Array:
        """
        Initialise environment food state

        Initialise empty state with a randomly placed rectangle of food

        Parameters
        ----------
        dims
            Environment dimensions
        key
            JAX random key

        Returns
        -------
        chex.Array
            Food state array
        """
        food = jnp.zeros(dims, dtype=float)
        food = self._drop_food(key, food)
        return food

    def update(self, key: chex.PRNGKey, step: int, food: chex.Array) -> chex.Array:
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
