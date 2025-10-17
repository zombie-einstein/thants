import abc
from typing import Sequence

import chex

from thants.generators.colonies.mono import ColonyGenerator
from thants.generators.colonies.utils import BBox, init_colonies
from thants.types import Colony


class ColoniesGenerator(abc.ABC):
    """
    Base generator of multiple colonies
    """

    def __init__(self, n_agents: Sequence[int], n_signals: int) -> None:
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
    def __call__(self, dims: tuple[int, int], key: chex.PRNGKey) -> Sequence[Colony]:
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
        Sequence[Colony]
            List of agent initial colony states
        """


class SingleColonyWrapper(ColoniesGenerator):
    """
    Wrapper class around a single colony generator
    """

    def __init__(self, generator: ColonyGenerator) -> None:
        """
        Initialise the wrapper

        Parameters
        ----------
        generator
            Single colony generator
        """
        self.generator = generator
        super().__init__([generator.n_agents], generator.n_signals)

    def __call__(self, dims: tuple[int, int], key: chex.PRNGKey) -> Sequence[Colony]:
        """
        Initialise the colony

        Parameters
        ----------
        dims
            Environment dimensions
        key
            JAX random key

        Returns
        -------
        Sequence[Colony]
            List of agent initial colony states
        """
        return [self.generator(dims, key)]


class DualBasicColoniesGenerator(ColoniesGenerator):
    """
    Basic generator that create 2 evenly spaced rectangular colonies
    """

    def __init__(
        self, n_agents: tuple[int, int], n_signals: int, nest_dims: tuple[int, int]
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
        super().__init__(n_agents, n_signals)

    def __call__(self, dims: tuple[int, int], key: chex.PRNGKey) -> Sequence[Colony]:
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
        Sequence[Colony]
            List of initialised colonies
        """
        mid = dims[1] // 2
        bounds = [
            BBox(x0=(0, 0), x1=(dims[0], mid)),
            BBox(x0=(0, mid), x1=dims),
        ]
        return init_colonies(
            dims, self.nest_dims, self.n_agents, self.n_signals, bounds
        )


class QuadBasicColoniesGenerator(ColoniesGenerator):
    """
    Basic generator that create 4 evenly spaced rectangular colonies
    """

    def __init__(
        self,
        n_agents: tuple[int, int, int, int],
        n_signals: int,
        nest_dims: tuple[int, int],
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
        super().__init__(n_agents, n_signals)

    def __call__(self, dims: tuple[int, int], key: chex.PRNGKey) -> Sequence[Colony]:
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
        Sequence[Colony]
            List of initialised colonies
        """

        mid = (dims[0] // 2, dims[1] // 2)
        bounds = [
            BBox(x0=(0, 0), x1=mid),
            BBox(x0=mid, x1=dims),
            BBox(x0=(mid[0], 0), x1=(dims[0], mid[1])),
            BBox(x0=(0, mid[1]), x1=(mid[0], dims[1])),
        ]
        return init_colonies(
            dims, self.nest_dims, self.n_agents, self.n_signals, bounds
        )
