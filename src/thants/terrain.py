import abc

import chex
import jax.numpy as jnp


class TerrainGenerator(abc.ABC):
    """
    Base terrain generation class
    """

    @abc.abstractmethod
    def __call__(self, dims: tuple[int, int], key: chex.PRNGKey) -> chex.Array:
        """
        Generate an array indicating passable/unpassable terrain

        Parameters
        ----------
        dims
            Environment dimensions
        key
            JAX random key

        Returns
        -------
        chex.Array
            Array of boolean values indicating if a cell is passable by ants
        """


class OpenTerrainGenerator(TerrainGenerator):
    """
    Generates completely open terrain
    """

    def __call__(self, dims: tuple[int, int], key: chex.PRNGKey) -> chex.Array:
        """
        Generate an array indicating passable/unpassable terrain

        Parameters
        ----------
        dims
            Environment dimensions
        key
            JAX random key

        Returns
        -------
        chex.Array
            Array of boolean values indicating if a cell is passable by ants
        """
        return jnp.ones(dims, dtype=bool)


class BoundedTerrainGenerator(TerrainGenerator):
    """
    Generate open terrain with an unpassable boundary
    """

    def __call__(self, dims: tuple[int, int], key: chex.PRNGKey) -> chex.Array:
        """
        Generate an array indicating passable/unpassable terrain

        Parameters
        ----------
        dims
            Environment dimensions
        key
            JAX random key

        Returns
        -------
        chex.Array
            Array of boolean values indicating if a cell is passable by ants
        """
        terrain = jnp.zeros(dims, dtype=bool)
        terrain = terrain.at[1 : dims[0] - 1, 1 : dims[1] - 1].set(True)
        return terrain
