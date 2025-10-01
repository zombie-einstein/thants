from functools import cached_property, partial
from typing import Optional, Sequence, Tuple

import chex
import jax
import jax.numpy as jnp
from jumanji import Environment, specs
from jumanji.types import TimeStep, restart, termination, transition
from jumanji.viewer import Viewer
from matplotlib.animation import FuncAnimation

from . import steps
from .actions import derive_actions
from .generator import BasicGenerator, Generator
from .observations import observations_from_state
from .rewards import NullRewardFn, RewardFn
from .signals import BasicSignalPropagator, SignalPropagator
from .types import Ants, Observations, State
from .viewer import ThantsViewer


class Thants(Environment):
    """
    Thants environment
    """

    def __init__(
        self,
        generator: Optional[Generator] = None,
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
        generator
            Initial state generator, initialises ants, food and nest values.
            By default, implements a `BasicGenerator` for a 100x100 space
            and 25 agents. The generator is also responsible for depositing
            new food during the simulation.
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
        self.carry_capacity = carry_capacity
        self.max_steps = max_steps
        self._signal_dynamics = signal_dynamics or BasicSignalPropagator(
            n_signals=2, decay_rate=0.002, dissipation_rate=0.2
        )
        self._generator = generator or BasicGenerator(
            (100, 100), 25, (5, 5), (5, 5), 100
        )
        self._reward_fn = reward_fn or NullRewardFn()
        self._viewer = viewer or ThantsViewer(self._generator.dims)
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
        key, init_key = jax.random.split(key, num=2)
        ants, nest, food = self._generator.init(init_key)
        signals = jnp.zeros(
            (self._signal_dynamics.n_signals, *self._generator.dims), dtype=float
        )
        state = State(
            step=0,
            key=key,
            ants=ants,
            food=food,
            signals=signals,
            nest=nest,
        )
        observations = observations_from_state(state)
        time_step = restart(observation=observations, shape=(self._generator.n_agents,))
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
        actions = derive_actions(actions)

        # Apply movements
        new_pos = steps.update_positions(
            self._generator.dims, state.ants.pos, actions.movements
        )

        # Pick up and drop-off food
        new_food, new_carrying = steps.update_food(
            state.food,
            new_pos,
            actions.take_food,
            actions.deposit_food,
            state.ants.carrying,
            self.carry_capacity,
        )
        # Drop any new food
        new_food = self._generator.update_food(food_key, state.step, new_food)
        # Propagate / disperse signals
        new_signals = self._signal_dynamics(signals_key, state.signals)
        # Deposit signals
        new_signals = steps.deposit_signals(
            new_signals, new_pos, actions.deposit_signals
        )

        new_state = State(
            step=state.step + 1,
            key=key,
            ants=Ants(pos=new_pos, health=state.ants.health, carrying=new_carrying),
            food=new_food,
            signals=new_signals,
            nest=state.nest,
        )
        observations = observations_from_state(new_state)
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
        return self._generator.n_agents

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
            shape=(self.num_agents, 9),
            minimum=0.0,
            maximum=1.0,
            dtype=float,
            name="ants",
        )
        food = specs.BoundedArray(
            shape=(self.num_agents, 9),
            minimum=0.0,
            maximum=jnp.inf,
            dtype=float,
            name="food",
        )
        signals = specs.BoundedArray(
            shape=(self.num_agents, self._signal_dynamics.n_signals, 9),
            minimum=0.0,
            maximum=jnp.inf,
            dtype=float,
            name="signals",
        )
        nest = specs.BoundedArray(
            shape=(self.num_agents, 9),
            minimum=0.0,
            maximum=1.0,
            dtype=float,
            name="nest",
        )
        carrying = specs.BoundedArray(
            shape=(self.num_agents,),
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
            shape=(self._generator.n_agents,),
            minimum=0,
            maximum=7 + self._signal_dynamics.n_signals,
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
            shape=(self._generator.n_agents,),
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
