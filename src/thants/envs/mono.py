from functools import cached_property
from typing import Optional, Sequence

import chex
from jumanji import Environment, specs
from jumanji.types import TimeStep
from jumanji.viewer import Viewer
from matplotlib.animation import FuncAnimation

from thants.envs.multi import Thants
from thants.generators.colonies.mono import (
    BasicColonyGenerator,
    ColonyGenerator,
)
from thants.generators.colonies.multi import SingleColonyWrapper
from thants.generators.food import FoodGenerator
from thants.generators.terrain import TerrainGenerator
from thants.rewards import RewardFn
from thants.signals import SignalPropagator
from thants.types import Observations, State


class ThantsMono(Environment):
    """Environment with a single colony"""

    def __init__(
        self,
        dims: tuple[int, int] = (50, 50),
        colony_generator: Optional[ColonyGenerator] = None,
        food_generator: Optional[FoodGenerator] = None,
        terrain_generator: Optional[TerrainGenerator] = None,
        signal_dynamics: Optional[SignalPropagator] = None,
        reward_fn: Optional[RewardFn] = None,
        viewer: Optional[Viewer[State]] = None,
        max_steps: int = 10_000,
        carry_capacity: float = 1.0,
        take_food_amount: float = 0.1,
        deposit_food_amount: float = 0.1,
        signal_deposit_amount: float = 0.1,
    ) -> None:
        """
        Initialise the environment

        Parameters
        ----------
        dims
            Environment grid dimensions, default is a 50x50 environment
        colony_generator
            Initial ant colony state generator, initialises ants and nest values.
            By default, initialises a `BasicColonyGenerator` with 25 ants, 2
            signal-channels and 5x5 nest.
        food_generator
            Food state generator and updater. By default, initialises a
            `BasicFoodGenerator` that creates 5x5 rectangular patches of food at
            fixed intervals.
        terrain_generator
            Terrain generator. By default, initialises a `OpenTerrainGenerator`
            that initialises a map with no obstacles.
        signal_dynamics
            Signal propagation functionality. By default, implements a
            `BasicSignalPropagator`.
        reward_fn
            Reward function, by default rewards are supplied when an ant drops food on
            the nest.
        viewer
            Environment visualiser. By default, initialises a viewer using a Matplotlib
            backend.
        max_steps
            Maximum environment steps
        carry_capacity
            Maximum ant carrying capacity
        take_food_amount
            Amount of (attempted) food taken by a take food action
        deposit_food_amount
            Amount of (attempted) food deposited by a deposit food action
        signal_deposit_amount
            Amount of signal deposited by the deposit signal action
        """
        colony_generator = colony_generator or BasicColonyGenerator(25, 2, (5, 5))
        colony_generator = SingleColonyWrapper(colony_generator)
        self.env = Thants(
            dims=dims,
            colonies_generator=colony_generator,
            food_generator=food_generator,
            terrain_generator=terrain_generator,
            signal_dynamics=signal_dynamics,
            reward_fn=reward_fn,
            viewer=viewer,
            max_steps=max_steps,
            carry_capacity=carry_capacity,
            take_food_amount=take_food_amount,
            deposit_food_amount=deposit_food_amount,
            signal_deposit_amount=signal_deposit_amount,
        )
        super().__init__()

    def reset(self, key: chex.PRNGKey) -> tuple[State, TimeStep[Observations]]:
        """
        Reset the environment state

        Parameters
        ----------
        key
            JAX random key

        Returns
        -------
        tuple[State, TimeStep]
            Tuple containing new environment state, and initial timestep
        """
        state, timestep = self.env.reset(key)
        return state, timestep[0]

    def step(
        self, state: State, actions: chex.Array
    ) -> tuple[State, TimeStep[Observations]]:
        """
        Update the state of the environment

        Update performs the following steps

        - Unwrap actions into state updates
        - Apply position updates
        - Apply food pick-up/deposit updates
        - Dissipate and propagate signals
        - Apply signal deposit actions
        - Clear any food returned to the nest

        Parameters
        ----------
        state
            Current environment state
        actions
            Integer array of individual ant actions

        Returns
        -------
        tuple[State, TimeStep]
            Tuple containing new state and TimeStep
        """
        state, timestep = self.env.step(state, [actions])
        return state, timestep[0]

    @cached_property
    def dims(self) -> tuple[int, int]:
        return self.env.dims

    @cached_property
    def num_agents(self) -> int:
        return self.env.num_agents[0]

    @cached_property
    def observation_spec(self) -> specs.Spec[Observations]:
        """
        Observation specification

        The observation consists of several components:

        - `[n-agents, 1, 9]` view of ants in the local vicinity
        - `[n-agents, 9]` view of food deposits in local vicinity
        - `[n-agents, n-signals, 9]` view of signals in the local vicinity
        - `[n-agents, 1, 9]` view indicating nest locations in the vicinity
        - `[n_agents,]` amount of food being carried by ants

        Returns
        -------
        ObservationSpec
        """
        return self.env.observation_spec[0]

    @cached_property
    def action_spec(self) -> specs.BoundedArray:
        """
        Action specification

        Actions are given by an array of integers indicating the discrete action
        to be taken by each ant.

        Returns
        -------
        ActionSpec
        """
        return self.env.action_spec[0]

    @cached_property
    def reward_spec(self) -> specs.Array:
        """
        Reward specification

        Array of individual rewards for each ant agent

        Returns
        -------
        RewardSpec
        """
        return self.env.reward_spec[0]

    def render(self, state: State) -> None:
        """Render a frame of the environment for a given state using matplotlib.

        Args:
            state: State object.
        """
        self.env.render(state)

    def animate(
        self,
        states: Sequence[State],
        interval: int = 100,
        save_path: Optional[str] = None,
    ) -> FuncAnimation:
        """Create an animation from a sequence of environment states.

        Args:
            states: sequence of environment states corresponding to consecutive
                timesteps.
            interval: delay between frames in milliseconds.
            save_path: the path where the animation file should be saved. If it
                is None, the plot will not be saved.

        Returns:
            Animation that can be saved as a GIF, MP4, or rendered with HTML.
        """
        return self.env.animate(states, interval=interval, save_path=save_path)

    def close(self) -> None:
        """Perform any necessary cleanup."""
        self.env.close()
