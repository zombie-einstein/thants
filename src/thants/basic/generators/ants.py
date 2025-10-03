import abc
import math

import chex
import jax.numpy as jnp

from thants.basic.types import Ants
from thants.common.utils import get_rectangular_indices


class AntGenerator(abc.ABC):
    """
    Base initial state generator and food updater
    """

    def __init__(self, n_agents: int) -> None:
        """
        Initialise base attributes

        Parameters
        ----------
        n_agents
            Number of ant agents to initialise
        """
        self.n_agents = n_agents

    @abc.abstractmethod
    def __call__(
        self, dims: tuple[int, int], key: chex.PRNGKey
    ) -> tuple[Ants, chex.Array]:
        """
        Initialise ants, and nest

        Parameters
        ----------
        dims
            Environment dimensions
        key
            JAX random key

        Returns
        -------
        tuple[Ants, chex.Array]
            Tuple containing ant, and nest states
        """


class BasicAntGenerator(AntGenerator):
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
        n_agents: int,
        nest_dims: tuple[int, int],
    ) -> None:
        """
        Initialise a basic generator

        Parameters
        ----------
        n_agents
            Number of ant agents to generate
        nest_dims
            Dimensions of the nest region
        """
        self.nest_dims = nest_dims
        super().__init__(n_agents)

    def __call__(
        self, dims: tuple[int, int], key: chex.PRNGKey
    ) -> tuple[Ants, chex.Array]:
        """
        Initialise ant and nest states

        Initialises a new initial state with

        - A rectangular nest at the centre of the environment
        - Ants placed in an approximate square at the centre of the environment

        Parameters
        ----------
        dims
            Environment dimensions
        key
            JAX random key

        Returns
        -------
        tuple[Ants, chex.Array]
            Tuple containing ant and nest states
        """
        assert self.n_agents <= dims[0] * dims[1]

        dims = jnp.array(dims)
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
        nest = jnp.zeros(dims, dtype=bool)
        nest = nest.at[nest_idxs[:, 0], nest_idxs[:, 1]].set(True)

        return ants, nest
