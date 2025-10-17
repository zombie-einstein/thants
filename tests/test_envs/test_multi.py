import chex
import jax
import jax.numpy as jnp
import pytest

from thants.envs.multi import Thants
from thants.generators.colonies.multi import DualBasicColoniesGenerator
from thants.generators.food import BasicFoodGenerator
from thants.types import Observations, State


@pytest.fixture
def env() -> Thants:
    dims = (20, 40)
    colony_generator = DualBasicColoniesGenerator((16, 9), 2, (5, 5))
    food_generator = BasicFoodGenerator(
        (5, 5),
        5,
    )
    return Thants(
        dims=dims, colonies_generator=colony_generator, food_generator=food_generator
    )


def test_env_does_not_smoke(key: chex.Array, env: Thants) -> None:
    """Test that we can run an episode without any errors."""
    env.max_steps = 100
    n_steps = 50

    def step(_state: State, _: None) -> tuple[State, list[Observations]]:
        k1, k2 = jax.random.split(_state.key, 2)
        actions = [
            jax.random.choice(k1, 9, (env.num_agents[0],)),
            jax.random.choice(k2, 9, (env.num_agents[1],)),
        ]
        _state, timesteps = env.step(_state, actions)
        return _state, [t.observation for t in timesteps]

    state, _ = env.reset(key)

    state, obs = jax.lax.scan(step, state, None, n_steps)

    assert isinstance(state, State)
    assert state.food.shape == env.dims
    assert state.terrain.shape == env.dims
    assert state.colonies.ants.pos.shape == (sum(env.num_agents), 2)
    assert state.colonies.nests.shape == env.dims
    assert jnp.all(
        jnp.logical_not(jnp.logical_and(state.food > 0.0, state.colonies.nests > 0))
    )

    assert isinstance(obs, list)
    assert len(obs) == 2
    assert all([isinstance(x, Observations) for x in obs])

    for n, o in zip(env.num_agents, obs):
        assert o.ants.shape == (n_steps, n, 2, 9)
