import abc

import chex

from thants.generators.colonies.utils import BBox, init_colony
from thants.types import Colony


class ColonyGenerator(abc.ABC):
    """
    Base initial colony state generator
    """

    def __init__(self, n_agents: int, n_signals: int) -> None:
        """
        Initialise base attributes

        Parameters
        ----------
        n_agents
            Number of ant agents to initialise in the colony
        """
        self.n_agents = n_agents
        self.n_signals = n_signals

    @abc.abstractmethod
    def __call__(self, dims: tuple[int, int], key: chex.PRNGKey) -> Colony:
        """
        Initialise colony state

        Parameters
        ----------
        dims
            Environment dimensions
        key
            JAX random key

        Returns
        -------
        Colony
            Colony state containing ants, signals, and nest states
        """


class BasicColonyGenerator(ColonyGenerator):
    """
    Basic generator that creates rectangular nest and initial ant positions

    Generator that places:

    - A rectangular nest at the centre of the environment
    - Ants in a square region at the centre of the environment
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
        n_signals
            Number of signal channels
        nest_dims
            Rectangular dimensions of the nest region
        """
        self.nest_dims = nest_dims
        super().__init__(n_agents, n_signals)

    def __call__(self, dims: tuple[int, int], key: chex.PRNGKey) -> Colony:
        """
        Initialise colony state

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
        Colony
            Colony state containing ants, signals, and nest states
        """
        bounds = BBox(x0=(0, 0), x1=dims)
        return init_colony(dims, bounds, self.nest_dims, self.n_agents, self.n_signals)
