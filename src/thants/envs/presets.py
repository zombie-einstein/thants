from thants.envs.multi import Thants
from thants.generators.colonies.multi import (
    DualBasicColoniesGenerator,
    QuadBasicColoniesGenerator,
)
from thants.generators.food import BasicFoodGenerator


class ThantsDual(Thants):
    """
    Environment with two evenly sized and spaced rectangular colonies
    """

    def __init__(
        self,
        dims: tuple[int, int] = (50, 100),
        n_agents: int = 36,
        n_signals: int = 2,
        nest_dims: tuple[int, int] = (5, 5),
        food_drop_dims: tuple[int, int] = (5, 5),
        food_drop_interval: int = 50,
        food_decay_rate: float = 0.0,
        max_steps: int = 10_000,
        carry_capacity: float = 1.0,
        take_food_amount: float = 0.1,
        deposit_food_amount: float = 0.1,
        signal_deposit_amount: float = 0.1,
        view_distance: int = 1,
    ) -> None:
        """
        Initialise the environment

        Parameters
        ----------
        dims
            Env dimensions
        n_agents
            Number of agents in each colony
        n_signals
            Number of signal channels
        nest_dims
            Rectangular dimensions of each colonies nest
        food_drop_dims
            Rectangular dimensions of food deposits
        food_drop_interval
            Interval between new randomly placed food deposits
        food_decay_rate
            Amount any depositied food is reduced each step
        max_steps
            Maximum environment steps
        carry_capacity
            Ant food carrying capacity
        take_food_amount
            Max food that can be picked up by an ant in a single step
        deposit_food_amount
            Max food that can be dropped by an ant in a single step
        signal_deposit_amount
            Amount of signal deposited in a single step
        view_distance
            Number of cells away from an agent observed by each agent
        """
        food_generator = BasicFoodGenerator(
            food_drop_dims, food_drop_interval, decay_rate=food_decay_rate
        )
        super().__init__(
            dims=dims,
            colonies_generator=DualBasicColoniesGenerator(
                n_agents=(n_agents, n_agents), n_signals=n_signals, nest_dims=nest_dims
            ),
            food_generator=food_generator,
            max_steps=max_steps,
            carry_capacity=carry_capacity,
            take_food_amount=take_food_amount,
            deposit_food_amount=deposit_food_amount,
            signal_deposit_amount=signal_deposit_amount,
            view_distance=view_distance,
        )


class ThantsQuad(Thants):
    """
    Environment with four evenly sized and spaced rectangular colonies
    """

    def __init__(
        self,
        dims: tuple[int, int] = (100, 100),
        n_agents: int = 36,
        n_signals: int = 2,
        nest_dims: tuple[int, int] = (5, 5),
        food_drop_dims: tuple[int, int] = (5, 5),
        food_drop_interval: int = 100,
        food_decay_rate: float = 0.0,
        max_steps: int = 10_000,
        carry_capacity: float = 1.0,
        take_food_amount: float = 0.1,
        deposit_food_amount: float = 0.1,
        signal_deposit_amount: float = 0.1,
        view_distance: int = 1,
    ) -> None:
        """
        Initialise the environment

        Parameters
        ----------
        dims
            Env dimensions
        n_agents
            Number of agents in each colony
        n_signals
            Number of signal channels
        nest_dims
            Rectangular dimensions of each colonies nest
        food_drop_dims
            Rectangular dimensions of food deposits
        food_drop_interval
            Interval between new randomly placed food deposits
        food_decay_rate
            Amount any depositied food is reduced each step
        max_steps
            Maximum environment steps
        carry_capacity
            Ant food carrying capacity
        take_food_amount
            Max food that can be picked up by an ant in a single step
        deposit_food_amount
            Max food that can be dropped by an ant in a single step
        signal_deposit_amount
            Amount of signal deposited in a single step
        view_distance
            Number of cells away from an agent observed by each agent
        """
        food_generator = BasicFoodGenerator(
            food_drop_dims, food_drop_interval, decay_rate=food_decay_rate
        )
        super().__init__(
            dims=dims,
            colonies_generator=QuadBasicColoniesGenerator(
                n_agents=(n_agents, n_agents, n_agents, n_agents),
                n_signals=n_signals,
                nest_dims=nest_dims,
            ),
            food_generator=food_generator,
            max_steps=max_steps,
            carry_capacity=carry_capacity,
            take_food_amount=take_food_amount,
            deposit_food_amount=deposit_food_amount,
            signal_deposit_amount=signal_deposit_amount,
            view_distance=view_distance,
        )
