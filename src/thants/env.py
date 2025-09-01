from typing import Tuple

import chex
import jax.numpy as jnp
from jumanji import Environment, specs
from jumanji.env import ActionSpec
from jumanji.types import TimeStep, restart, transition

from . import steps
from .types import Ants, Observation, State


class Thants(Environment):
    def __init__(
        self,
        dims: tuple[int, int],
        n_agents: int,
        decay_rate=0.05,
        dissipation_rate=0.0,
    ):
        self.dims = dims
        self.n_agents = n_agents
        self.decay_rate = decay_rate
        self.dissipation_rate = dissipation_rate
        super().__init__()

    def reset(self, key: chex.PRNGKey) -> Tuple[State, TimeStep[Observation]]:
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
        observation = Observation(local=jnp.zeros((self.n_agents, 9)))
        time_step = restart(observation=observation, shape=(self.n_agents,))
        return state, time_step

    def step(
        self, state: State, action: chex.Array
    ) -> Tuple[State, TimeStep[Observation]]:
        # Unwrap actions
        movement_actions = jnp.zeros((self.n_agents, 2), dtype=int)
        take_food_actions = jnp.zeros((self.n_agents,))
        deposit_food_actions = jnp.zeros((self.n_agents,))
        deposit_signal_actions = jnp.zeros((self.n_agents,))

        # Apply movements
        new_pos = steps.update_positions(self.dims, state.ants.pos, movement_actions)
        # Pick up and drop-off food
        new_food, new_carrying = steps.update_food(
            state.food,
            new_pos,
            take_food_actions,
            deposit_food_actions,
            state.ants.carrying,
            1.0,
        )
        # Dissipate chemicals
        # Decay chemicals
        new_signals = steps.update_signals(
            state.signals, self.dissipation_rate, self.decay_rate
        )
        # Deposit signals
        new_signals = steps.deposit_signals(
            new_signals, new_pos, deposit_signal_actions
        )

        new_state = State(
            ants=Ants(pos=new_pos, health=state.ants.health, carrying=new_carrying),
            food=new_food,
            signals=new_signals,
            nest=state.nest,
        )
        observations = Observation(local=jnp.zeros((self.n_agents, 9)))
        rewards = jnp.zeros((self.n_agents,))
        timestep = transition(rewards, observations, shape=(self.n_agents,))
        return new_state, timestep

    def observation_spec(self) -> specs.Spec[Observation]:
        pass

    def action_spec(self) -> ActionSpec:
        pass
