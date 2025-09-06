from functools import cached_property
from typing import Optional, Tuple

import chex
import jax.numpy as jnp
from jumanji import Environment, specs
from jumanji.types import TimeStep, restart, transition

from . import steps
from .actions import derive_actions
from .observations import observations_from_state
from .rewards import NullRewardFn, RewardFn
from .types import Ants, Observations, State


class Thants(Environment):
    def __init__(
        self,
        dims: tuple[int, int],
        n_agents: int,
        decay_rate=0.05,
        dissipation_rate=0.0,
        reward_fn: Optional[RewardFn] = None,
    ) -> None:
        self.dims = dims
        self.n_agents = n_agents
        self.decay_rate = decay_rate
        self.dissipation_rate = dissipation_rate
        self.reward_fn = reward_fn or NullRewardFn()
        self.carry_capacity = 1.0
        super().__init__()

    def reset(self, key: chex.PRNGKey) -> Tuple[State, TimeStep[Observations]]:
        food = jnp.zeros(self.dims, dtype=int)
        signals = jnp.zeros(self.dims, dtype=float)
        nest = jnp.zeros(self.dims, dtype=bool)
        ant_pos = jnp.indices((5, 5)).reshape((25, 2))
        ant_health = jnp.ones((self.n_agents,))
        ant_carrying = jnp.zeros((self.n_agents,))
        state = State(
            ants=Ants(
                pos=ant_pos,
                health=ant_health,
                carrying=ant_carrying,
            ),
            food=food,
            signals=signals,
            nest=nest,
        )
        observations = observations_from_state(self.dims, state)
        time_step = restart(observation=observations, shape=(self.n_agents,))
        return state, time_step

    def step(
        self, state: State, actions: chex.Array
    ) -> Tuple[State, TimeStep[Observations]]:
        # Unwrap actions
        actions = derive_actions(actions)

        # Apply movements
        new_pos = steps.update_positions(self.dims, state.ants.pos, actions.movements)

        # Pick up and drop-off food
        new_food, new_carrying = steps.update_food(
            state.food,
            new_pos,
            actions.take_food,
            actions.deposit_food,
            state.ants.carrying,
            self.carry_capacity,
        )
        # Dissipate chemicals
        new_signals = steps.update_signals(
            state.signals, self.dissipation_rate, self.decay_rate
        )
        # Deposit signals
        new_signals = steps.deposit_signals(
            new_signals, new_pos, actions.deposit_signals
        )

        new_state = State(
            ants=Ants(pos=new_pos, health=state.ants.health, carrying=new_carrying),
            food=new_food,
            signals=new_signals,
            nest=state.nest,
        )
        observations = observations_from_state(self.dims, new_state)
        rewards = self.reward_fn(old_state=state, new_state=new_state)
        timestep = transition(rewards, observations, shape=(self.n_agents,))
        return new_state, timestep

    @cached_property
    def num_agents(self) -> int:
        return self.n_agents

    @cached_property
    def observation_spec(self) -> specs.Spec[Observations]:
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
            shape=(self.num_agents, 9),
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
        return specs.BoundedArray(
            shape=(self.n_agents,),
            minimum=0,
            maximum=8,
            dtype=int,
        )

    @cached_property
    def reward_spec(self) -> specs.Array:
        return specs.Array(
            shape=(self.n_agents,),
            dtype=float,
        )
