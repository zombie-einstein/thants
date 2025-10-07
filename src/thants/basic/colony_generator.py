import abc

import chex

from thants.common.types import Colony
from thants.common.utils import init_colony


class ColonyGenerator(abc.ABC):
    """
    Base initial state generator and food updater
    """

    def __init__(self, n_agents: int, n_signals: int) -> None:
        """
        Initialise base attributes

        Parameters
        ----------
        n_agents
            Number of ant agents to initialise
        """
        self.n_agents = n_agents
        self.n_signals = n_signals

    @abc.abstractmethod
    def __call__(self, dims: tuple[int, int], key: chex.PRNGKey) -> Colony:
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


class BasicColonyGenerator(ColonyGenerator):
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
        n_signals: int,
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
        super().__init__(n_agents, n_signals)

    def __call__(self, dims: tuple[int, int], key: chex.PRNGKey) -> Colony:
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
        return init_colony(
            dims, (0, 0), dims, self.nest_dims, self.n_agents, self.n_signals
        )
