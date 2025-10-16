from functools import cached_property, partial
from typing import Optional, Sequence

import chex
import jax
import jax.numpy as jnp
from jumanji import Environment, specs
from jumanji.types import TimeStep, restart, termination, transition
from jumanji.viewer import Viewer
from matplotlib.animation import FuncAnimation

from thants.actions import derive_actions
from thants.generators.colonies.multi import (
    ColoniesGenerator,
    DualBasicColoniesGenerator,
)
from thants.generators.food import BasicFoodGenerator, FoodGenerator
from thants.generators.terrain import OpenTerrainGenerator, TerrainGenerator
from thants.observations import observations_from_state
from thants.rewards import DeliveredFoodRewards, RewardFn
from thants.signals import BasicSignalPropagator, SignalPropagator
from thants.specs import get_action_spec, get_observation_spec, get_reward_spec
from thants.steps import (
    clear_nest,
    deposit_signals,
    merge_colonies,
    update_food,
    update_positions,
)
from thants.types import Ants, Colonies, Observations, State
from thants.viewer import ThantsMultiColonyViewer


class Thants(Environment):
    """
    Thants environment with multiple colonies
    """

    def __init__(
        self,
        dims: tuple[int, int] = (50, 100),
        colonies_generator: Optional[ColoniesGenerator] = None,
        food_generator: Optional[FoodGenerator] = None,
        terrain_generator: Optional[TerrainGenerator] = None,
        signal_dynamics: Optional[SignalPropagator] = None,
        reward_fn: Optional[RewardFn] = None,
        viewer: Optional[Viewer[State]] = None,
        max_steps: int = 1_000,
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
            Environment grid dimensions, default is a 50x100 environment
        colonies_generator
            Initial colonies state generator, initialises ants and nest states.
            By default, initialises a `BasicColoniesGenerator` with 2 colonies,
            with 25 agents each, and 2 signal channels.
        food_generator
            Food initial state generator and updator. By default, initialises a
            `BasicFoodGenerator` that creates 5x5 rectangular patches of food at
            fixed intervals.
        terrain_generator
            Terrain generator. By default, initialises a `OpenTerrainGenerator`
            that initialises a map with no obstacles.
        signal_dynamics
            Signal propagation functionality. By default, implements a
            `BasicSignalPropagator` with 2 signal channels.
        reward_fn
            Reward function, by default rewards agents for depositing food on
            their own nest.
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
        self._colonies_generator = colonies_generator or DualBasicColoniesGenerator(
            [25, 25], 2, (5, 5)
        )
        self._food_generator = food_generator or BasicFoodGenerator((5, 5), 100, 1.0)
        self._terrain_generator = terrain_generator or OpenTerrainGenerator()
        self._signal_dynamics = signal_dynamics or BasicSignalPropagator(
            decay_rate=0.002, dissipation_rate=0.2
        )
        self._reward_fn = reward_fn or DeliveredFoodRewards()
        self._viewer = viewer or ThantsMultiColonyViewer()
        super().__init__()

    def reset(
        self, key: chex.PRNGKey
    ) -> tuple[State, Sequence[TimeStep[Observations]]]:
        """
        Reset the environment state

        Parameters
        ----------
        key
            JAX random key

        Returns
        -------
        tuple[State, Sequence[TimeStep]]
            Tuple containing new environment state, and initial timesteps
            for each colony
        """
        key, colony_key, food_key, terrain_key = jax.random.split(key, num=4)
        colonies = self._colonies_generator(self.dims, colony_key)
        colonies = merge_colonies(colonies)
        food = self._food_generator.init(self.dims, food_key)
        terrain = self._terrain_generator(self.dims, terrain_key)
        state = State(
            step=0,
            key=key,
            colonies=colonies,
            food=food,
            terrain=terrain,
        )
        observations = observations_from_state(self.num_agents, state)
        time_steps = [
            restart(observation=obs, shape=(n,))
            for obs, n in zip(observations, self._colonies_generator.n_agents)
        ]
        return state, time_steps

    def step(
        self, state: State, actions: Sequence[chex.Array]
    ) -> tuple[State, Sequence[TimeStep[Observations]]]:
        """
        Update the state of the environment

        Update performs the following steps

        - Unwrap actions into state updates
        - Apply position updates
        - Apply food pick-up/deposit updates
        - Dissipate and propagate signals
        - Apply signal deposit actions
        - Clear any food deposited on nests

        Parameters
        ----------
        state
            Current environment state
        actions
            List of action arrays for each colony

        Returns
        -------
        tuple[State, list[TimeStep]]
            Tuple containing new state and list of TimeSteps for each colony
        """
        key, food_key, signals_key = jax.random.split(state.key, num=3)
        actions = jnp.concatenate(actions, axis=0)
        # Unwrap actions
        actions = derive_actions(
            actions,
            take_food_amount=self.take_food_amount,
            deposit_food_amount=self.deposit_food_amount,
            signal_deposit_amount=self.signal_deposit_amount,
        )

        # Apply movements
        new_pos = update_positions(
            self.dims,
            state.colonies.ants.pos,
            state.terrain,
            actions.movements,
        )
        # Pick up and drop-off food for each colony
        new_food, new_carrying = update_food(
            state.food,
            new_pos,
            actions.take_food,
            actions.deposit_food,
            state.colonies.ants.carrying,
            self.carry_capacity,
        )
        # Drop any new food
        new_food = self._food_generator.update(food_key, state.step, new_food)
        # Propagate / disperse signals
        new_signals = self._signal_dynamics(signals_key, state.colonies.signals)
        # Deposit signals
        new_signals = deposit_signals(
            new_signals, new_pos, state.colonies.colony_idx, actions.deposit_signals
        )
        # Clear food dropped on nests
        new_food = clear_nest(state.colonies.nests, new_food)
        # Gather updated state
        colonies = Colonies(
            ants=Ants(
                pos=new_pos, health=state.colonies.ants.health, carrying=new_carrying
            ),
            colony_idx=state.colonies.colony_idx,
            signals=new_signals,
            nests=state.colonies.nests,
        )
        new_state = State(
            step=state.step + 1,
            key=key,
            colonies=colonies,
            food=new_food,
            terrain=state.terrain,
        )
        # Rewards
        rewards = self._reward_fn(self.num_agents, old_state=state, new_state=new_state)
        # Observations
        observations = observations_from_state(self.num_agents, new_state)
        timestep = [
            jax.lax.cond(
                state.step >= self.max_steps,
                partial(termination, shape=(n,)),
                partial(transition, shape=(n,)),
                rew,
                obs,
            )
            for rew, obs, n in zip(rewards, observations, self.num_agents)
        ]
        return new_state, timestep

    @cached_property
    def num_agents(self) -> Sequence[int]:
        return self._colonies_generator.n_agents

    @cached_property
    def num_colonies(self) -> int:
        return len(self.num_agents)

    @cached_property
    def num_actions(self) -> int:
        return 7 + self._colonies_generator.n_signals

    @cached_property
    def observation_spec(self) -> Sequence[specs.Spec[Observations]]:
        """
        List of observation specifications for each colony

        The observation consists of several components:

        - `[n-agents, 9]` view of ants in the local vicinity
        - `[n-agents, 9]` view of food deposits in local vicinity
        - `[n-agents, n-signals, 9]` view of signals in the local vicinity
        - `[n-agents, n-signals, 9]` view indicating nest locations in the vicinity
        - `[n_agents,]` amount of food being carried by ants

        Returns
        -------
        Sequence[ObservationSpec]
            Sequence of observation specs for each colony
        """
        return [
            get_observation_spec(
                n,
                self._colonies_generator.n_signals,
                self.carry_capacity,
                num_colonies=self.num_colonies,
            )
            for n in self.num_agents
        ]

    @cached_property
    def action_spec(self) -> Sequence[specs.BoundedArray]:
        """
        List of action specifications for each colony

        Actions for each colony are given by an array of integers indicating the
        discrete action to be taken by each ant.

        Returns
        -------
        Sequence[ActionSpec]
        """
        return [
            get_action_spec(n, self._colonies_generator.n_signals)
            for n in self.num_agents
        ]

    @cached_property
    def reward_spec(self) -> Sequence[specs.Array]:
        """
        List of reward specifications for each colony

        Array of rewards for each ant agent

        Returns
        -------
        Sequence[RewardSpec]
        """
        return [get_reward_spec(n) for n in self.num_agents]

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
