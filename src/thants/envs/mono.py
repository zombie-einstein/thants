from functools import cached_property
from typing import Optional, Sequence

import chex
from jumanji import Environment, specs
from jumanji.types import TimeStep
from jumanji.viewer import Viewer
from matplotlib.animation import FuncAnimation

from thants.envs.multi import ThantsMultiColony
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


class ThantsMonoColony(Environment):
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
        colony_generator = colony_generator or BasicColonyGenerator(25, 2, (5, 5))
        colony_generator = SingleColonyWrapper(colony_generator)
        self.env = ThantsMultiColony(
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
        state, timestep = self.env.reset(key)
        return state, timestep[0]

    def step(
        self, state: State, actions: chex.Array
    ) -> tuple[State, TimeStep[Observations]]:
        state, timestep = self.env.step(state, [actions])
        return state, timestep[0]

    @cached_property
    def num_agents(self) -> int:
        return self.env.num_agents[0]

    @cached_property
    def observation_spec(self) -> specs.Spec[Observations]:
        return self.env.observation_spec[0]

    @cached_property
    def action_spec(self) -> specs.BoundedArray:
        return self.env.action_spec[0]

    @cached_property
    def reward_spec(self) -> specs.Array:
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
        self.env._viewer.close()
