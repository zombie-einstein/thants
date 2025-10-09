from functools import cached_property, partial
from typing import Optional, Sequence, Tuple

import chex
import jax
import jax.numpy as jnp
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
from thants.common.steps import deposit_signals
from thants.common.types import Ants, Colony, Observations
from thants.multi.colonies_generator import (
    BasicColoniesGenerator,
    ColoniesGenerator,
)
from thants.multi.observations import observations_from_state
from thants.multi.steps import update_food, update_positions
from thants.multi.types import State
from thants.multi.viewer import ThantsMultiColonyViewer

from .rewards import NullRewardFn, RewardFn


class ThantsMultiColony(Environment):
    """
    Thants environment with dueling colonies
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
    ) -> None:
        """
        Initialise the environment

        Parameters
        ----------
        dims
            Environment grid dimensions
        colonies_generator
            Initial ant state generator, initialises ants and nest values.
            By default, initialises a `BasicAntGenerator` for a 100x100 space
            and 25 agents.
        food_generator
            Initial food state generator. By default, initialises a `BasicFoodGenerator`
            that creates 5x5 rectangular patches of food at fixed intervals.
        terrain_generator
            Terrain generator. By default, initialises a `OpenTerrainGenerator`
            that initialises a map with no obstacles.
        signal_dynamics
            Signal propagation functionality
            By default, implements a `BasicSignalPropagator` with 2 signal values.
        reward_fn
            Reward function, by default implements a default function that assigns
            0 rewards to all agents.
        viewer
            Environment visualiser. By default, initialises a viewer using a Matplotlib
            backend.
        max_steps
            Maximum environment steps
        carry_capacity
            Maximum ant carrying capacity
        """
        self.dims = dims
        self.carry_capacity = carry_capacity
        self.max_steps = max_steps
        self._colonies_generator = colonies_generator or BasicColoniesGenerator(
            25, 2, (5, 5)
        )
        self._food_generator = food_generator or BasicFoodGenerator((5, 5), 100, 1.0)
        self._terrain_generator = terrain_generator or OpenTerrainGenerator()
        self._signal_dynamics = signal_dynamics or BasicSignalPropagator(
            decay_rate=0.002, dissipation_rate=0.2
        )
        self._reward_fn = reward_fn or NullRewardFn()
        self._viewer = viewer or ThantsMultiColonyViewer()
        super().__init__()

    def reset(self, key: chex.PRNGKey) -> Tuple[State, list[TimeStep[Observations]]]:
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
        colonies = self._colonies_generator(self.dims, colony_key)
        food = self._food_generator.init(self.dims, food_key)
        terrain = self._terrain_generator(self.dims, terrain_key)
        state = State(
            step=0,
            key=key,
            colonies=colonies,
            food=food,
            terrain=terrain,
        )
        observations = observations_from_state(state)
        time_steps = [
            restart(observation=obs, shape=(n,))
            for obs, n in zip(observations, self._colonies_generator.n_agents)
        ]
        return state, time_steps

    def step(
        self, state: State, actions: list[chex.Array]
    ) -> Tuple[State, list[TimeStep[Observations]]]:
        """
        Update the state of the environment

        Update performs the following steps

        - Unwrap actions into state updates
        - Apply position updates
        - Apply food pick-up/deposit updates
        - Dissipate and propagate signals
        - Apply signal deposit actions

        Parameters
        ----------
        state
            Current environment state
        actions
            Array of individual ant actions

        Returns
        -------
        tuple[State, TimeStep]
            Tuple containing new state and TimeStep
        """
        key, food_key, signals_key = jax.random.split(state.key, num=3)
        # Unwrap actions
        actions = [derive_actions(a) for a in actions]

        # Apply movements
        new_pos = update_positions(
            self.dims,
            [c.ants.pos for c in state.colonies],
            state.terrain,
            [a.movements for a in actions],
        )
        # Pick up and drop-off food for each colony
        new_food, new_carrying = update_food(
            state.food,
            new_pos,
            [a.take_food for a in actions],
            [a.deposit_food for a in actions],
            [c.ants.carrying for c in state.colonies],
            self.carry_capacity,
        )
        # Drop any new food
        new_food = self._food_generator.update(food_key, state.step, new_food)
        # Propagate / disperse signals
        new_signals = [
            self._signal_dynamics(signals_key, c.signals) for c in state.colonies
        ]
        # Deposit signals
        new_signals = [
            deposit_signals(signals, pos, a.deposit_signals)
            for signals, pos, a in zip(new_signals, new_pos, actions)
        ]

        colonies = [
            Colony(
                ants=Ants(pos=pos, health=c.ants.health, carrying=carrying),
                signals=signals,
                nest=c.nest,
            )
            for c, pos, signals, carrying in zip(
                state.colonies, new_pos, new_signals, new_carrying
            )
        ]

        new_state = State(
            step=state.step + 1,
            key=key,
            colonies=colonies,
            food=new_food,
            terrain=state.terrain,
        )
        observations = observations_from_state(new_state)
        rewards = self._reward_fn(old_state=state, new_state=new_state)

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
    def num_agents(self) -> list[int]:
        return self._colonies_generator.n_agents

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
        ants = specs.BoundedArray(
            shape=(self.num_agents[0], 9),
            minimum=0.0,
            maximum=1.0,
            dtype=float,
            name="ants",
        )
        food = specs.BoundedArray(
            shape=(self.num_agents[0], 9),
            minimum=0.0,
            maximum=jnp.inf,
            dtype=float,
            name="food",
        )
        signals = specs.BoundedArray(
            shape=(self.num_agents[0], self._colonies_generator.n_signals, 9),
            minimum=0.0,
            maximum=jnp.inf,
            dtype=float,
            name="signals",
        )
        nest = specs.BoundedArray(
            shape=(self.num_agents[0], 9),
            minimum=0.0,
            maximum=1.0,
            dtype=float,
            name="nest",
        )
        terrain = specs.BoundedArray(
            shape=(self.num_agents[0], 9),
            minimum=0.0,
            maximum=1.0,
            dtype=float,
            name="terrain",
        )
        carrying = specs.BoundedArray(
            shape=(self.num_agents[0],),
            minimum=0.0,
            maximum=self.carry_capacity,
            dtype=float,
            name="carrying",
        )

        return specs.Spec(
            Observations,
            "ObservationSpec",
            ants=ants,
            food=food,
            signals=signals,
            nest=nest,
            carrying=carrying,
            terrain=terrain,
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
        return specs.BoundedArray(
            shape=(self._colonies_generator.n_agents[0],),
            minimum=0,
            maximum=7 + self._colonies_generator.n_signals,
            dtype=int,
        )

    @cached_property
    def reward_spec(self) -> specs.Array:
        """
        Reward specification

        Array of rewards for each ant agent

        Returns
        -------
        RewardSpec
        """
        return specs.Array(
            shape=(self._colonies_generator.n_agents[0],),
            dtype=float,
        )

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
