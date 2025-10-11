import abc

import chex

from thants.common.types import Colony
from thants.common.utils import init_colony


class ColoniesGenerator(abc.ABC):
    """
    Base generator of multiple colonies
    """

    def __init__(self, n_agents: list[int], n_signals: int) -> None:
        """
        Initialise base attributes

        Parameters
        ----------
        n_agents
            List of colony sizes
        n_signals
            Number of signal channels
        """
        self.n_signals = n_signals
        self.n_agents = n_agents

    @abc.abstractmethod
    def __call__(self, dims: tuple[int, int], key: chex.PRNGKey) -> list[Colony]:
        """
        Initialise colonies

        Parameters
        ----------
        dims
            Environment dimensions
        key
            JAX random key

        Returns
        -------
        list[Colony]
            List of agent initial colony states
        """


class BasicColoniesGenerator(ColoniesGenerator):
    """
    Basic generator that create 2 evenly spaced colonies
    """

    def __init__(
        self, n_agents: int, n_signals: int, nest_dims: tuple[int, int]
    ) -> None:
        """
        Initialise a basic generator

        Parameters
        ----------
        n_agents
            Number of agents in each colony
        n_signals
            Number of colony signal-channels
        nest_dims
            Rectangular nest dimensions
        """
        self.nest_dims = nest_dims
        super().__init__([n_agents, n_agents], n_signals)

    def __call__(self, dims: tuple[int, int], key: chex.PRNGKey) -> list[Colony]:
        """
        Initialise the pair of colonies

        Parameters
        ----------
        dims
            Dimensions of the environment
        key
            JAX random key

        Returns
        -------
        list[Colony]
            List of initialised colonies
        """
        mid = dims[1] // 2
        return [
            init_colony(
                dims,
                (0, 0),
                (dims[0], mid),
                self.nest_dims,
                self.n_agents[0],
                self.n_signals,
            ),
            init_colony(
                dims,
                (0, mid),
                (dims[0], dims[1]),
                self.nest_dims,
                self.n_agents[1],
                self.n_signals,
            ),
        ]
