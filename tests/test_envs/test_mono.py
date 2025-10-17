import chex
import jax
import jax.numpy as jnp
import pytest
from jumanji.testing.env_not_smoke import (
    check_env_does_not_smoke,
    check_env_specs_does_not_smoke,
)

from thants.envs.mono import ThantsMono
from thants.generators.colonies.mono import BasicColonyGenerator
from thants.generators.food import BasicFoodGenerator
from thants.types import Observations, State


@pytest.fixture
def env() -> ThantsMono:
    dims = (20, 20)
    colony_generator = BasicColonyGenerator(16, 2, (5, 5))
    food_generator = BasicFoodGenerator(
        (5, 5),
        5,
    )
    return ThantsMono(
        dims=dims, colony_generator=colony_generator, food_generator=food_generator
    )


def test_env_does_not_smoke(env: ThantsMono) -> None:
    """Test that we can run an episode without any errors."""
    env.max_steps = 20

    def select_action(action_key: chex.PRNGKey, _state: Observations) -> chex.Array:
        return jax.random.choice(action_key, 9, (env.num_agents,))

    check_env_does_not_smoke(env, select_action=select_action)


def test_env_specs_do_not_smoke(env: ThantsMono) -> None:
    """Test that we can access specs without any errors."""
    check_env_specs_does_not_smoke(env)


def state_checks(env: ThantsMono, state: State) -> None:
    assert isinstance(state, State)
    assert state.food.shape == env.dims
    assert state.terrain.shape == env.dims
    assert state.colonies.ants.pos.shape == (env.num_agents, 2)
    assert state.colonies.nests.shape == env.dims
    assert jnp.all(
        jnp.logical_not(jnp.logical_and(state.food > 0.0, state.colonies.nests > 0))
    )


def test_env_outputs(key: chex.Array, env: ThantsMono) -> None:
    state, obs = env.reset(key)

    state_checks(env, state)

    actions = jnp.zeros((env.num_agents,), dtype=int)
    state, obs = env.step(state, actions)

    state_checks(env, state)
