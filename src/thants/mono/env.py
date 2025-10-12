from functools import cached_property, partial
from typing import Optional, Sequence, Tuple

import chex
import jax
from jumanji import Environment, specs
from jumanji.types import TimeStep, restart, termination, transition
from jumanji.viewer import Viewer
from matplotlib.animation import FuncAnimation

from thants.common.actions import derive_actions
from thants.common.generators.food import BasicFoodGenerator, FoodGenerator
from thants.common.generators.terrain import (
    OpenTerrainGenerator,
    TerrainGenerator,
)
from thants.common.signals import BasicSignalPropagator, SignalPropagator
from thants.common.specs import (
    get_action_spec,
    get_observation_spec,
    get_reward_spec,
)
from thants.common.steps import deposit_signals, update_food
from thants.common.types import Ants, Colony, Observations
from thants.mono.colony_generator import BasicColonyGenerator, ColonyGenerator
from thants.mono.observations import observations_from_state
from thants.mono.rewards import DeliveredFoodRewards, RewardFn
from thants.mono.steps import clear_nest, update_positions
from thants.mono.types import State
from thants.mono.viewer import ThantsViewer


class ThantsMonoColony(Environment):
    """
    Thants single-colony environment
    """

    def __init__(
        self,
        dims: tuple[int, int],
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
            Environment grid dimensions
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
        self.dims = dims
        self.carry_capacity = carry_capacity
        self.take_food_amount = take_food_amount
        self.deposit_food_amount = deposit_food_amount
        self.signal_deposit_amount = signal_deposit_amount
        self.max_steps = max_steps
        self._colony_generator = colony_generator or BasicColonyGenerator(25, 2, (5, 5))
        self._food_generator = food_generator or BasicFoodGenerator((5, 5), 100, 1.0)
        self._terrain_generator = terrain_generator or OpenTerrainGenerator()
        self._signal_dynamics = signal_dynamics or BasicSignalPropagator(
            decay_rate=0.002, dissipation_rate=0.2
        )
        self._reward_fn = reward_fn or DeliveredFoodRewards()
        self._viewer = viewer or ThantsViewer()
        super().__init__()

    def reset(self, key: chex.PRNGKey) -> Tuple[State, TimeStep[Observations]]:
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
        key, colony_key, food_key, terrain_key = jax.random.split(key, num=4)
        colony = self._colony_generator(self.dims, colony_key)
        food = self._food_generator.init(self.dims, food_key)
        # For safety clear any food placed on a nest
        food = clear_nest(colony.nest, food)
        terrain = self._terrain_generator(self.dims, terrain_key)
        state = State(
            step=0,
            key=key,
            colony=colony,
            food=food,
            terrain=terrain,
        )
        observations = observations_from_state(state)
        time_step = restart(
            observation=observations, shape=(self._colony_generator.n_agents,)
        )
        return state, time_step

    def step(
        self, state: State, actions: chex.Array
    ) -> Tuple[State, TimeStep[Observations]]:
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
        key, food_key, signals_key = jax.random.split(state.key, num=3)

        # Unwrap actions
        actions = derive_actions(
            actions,
            take_food_amount=self.take_food_amount,
            deposit_food_amount=self.deposit_food_amount,
            signal_deposit_amount=self.signal_deposit_amount,
        )

        # Apply movements
        new_pos = update_positions(
            self.dims, state.colony.ants.pos, state.terrain, actions.movements
        )

        # Pick up and drop-off food
        new_food, new_carrying = update_food(
            state.food,
            new_pos,
            actions.take_food,
            actions.deposit_food,
            state.colony.ants.carrying,
            self.carry_capacity,
        )
        # Drop any new food
        new_food = self._food_generator.update(food_key, state.step, new_food)
        # Propagate / disperse signals
        new_signals = self._signal_dynamics(signals_key, state.colony.signals)
        # Deposit signals
        new_signals = deposit_signals(new_signals, new_pos, actions.deposit_signals)
        # Clear food dropped on the nest
        new_food = clear_nest(state.colony.nest, new_food)
        # Construct new state
        ants = Ants(pos=new_pos, health=state.colony.ants.health, carrying=new_carrying)
        colony = Colony(ants=ants, signals=new_signals, nest=state.colony.nest)
        new_state = State(
            step=state.step + 1,
            key=key,
            colony=colony,
            food=new_food,
            terrain=state.terrain,
        )
        # Observations
        observations = observations_from_state(new_state)
        # Rewards
        rewards = self._reward_fn(old_state=state, new_state=new_state)
        timestep = jax.lax.cond(
            state.step >= self.max_steps,
            partial(termination, shape=(self.num_agents,)),
            partial(transition, shape=(self.num_agents,)),
            rewards,
            observations,
        )
        return new_state, timestep

    @cached_property
    def num_agents(self) -> int:
        return self._colony_generator.n_agents

    @cached_property
    def num_actions(self) -> int:
        return 7 + self._colony_generator.n_signals

    @cached_property
    def observation_spec(self) -> specs.Spec[Observations]:
        """
        Observation specification

        The observation consists of several components:

        - `[n-agents, 9]` view of ants in the local vicinity
        - `[n-agents, 9]` view of food deposits in local vicinity
        - `[n-agents, n-signals, 9]` view of signals in the local vicinity
        - `[n-agents, n-signals, 9]` view indicating nest locations in the vicinity
        - `[n_agents,]` amount of food being carried by ants

        Returns
        -------
        ObservationSpec
        """
        return get_observation_spec(
            self.num_agents, self._colony_generator.n_signals, self.carry_capacity
        )

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
        return get_action_spec(self.num_agents, self._colony_generator.n_signals)

    @cached_property
    def reward_spec(self) -> specs.Array:
        """
        Reward specification

        Array of rewards for each ant agent

        Returns
        -------
        RewardSpec
        """
        return get_reward_spec(self.num_agents)

    def render(self, state: State) -> None:
        """Render a frame of the environment for a given state using matplotlib.

        Args:
            state: State object.
        """
        self._viewer.render(state)

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
        return self._viewer.animate(states, interval=interval, save_path=save_path)

    def close(self) -> None:
        """Perform any necessary cleanup."""
        self._viewer.close()
