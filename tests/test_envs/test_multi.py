import chex
import jax.random
import pytest

from thants.envs.multi import ThantsMultiColony
from thants.generators.colonies.multi import BasicColoniesGenerator
from thants.generators.food import BasicFoodGenerator
from thants.types import Observations, State


@pytest.fixture
def env() -> ThantsMultiColony:
    dims = (50, 100)
    colony_generator = BasicColoniesGenerator(64, 2, (5, 5))
    food_generator = BasicFoodGenerator(
        (2, 2),
        50,
    )
    return ThantsMultiColony(
        dims=dims, colonies_generator=colony_generator, food_generator=food_generator
    )


def test_env_does_not_smoke(key: chex.Array, env: ThantsMultiColony) -> None:
    """Test that we can run an episode without any errors."""
    env.max_steps = 100

    def step(_state: State, _: None) -> tuple[State, list[Observations]]:
        k1, k2 = jax.random.split(_state.key, 2)
        actions = [
            jax.random.choice(k1, 9, (env.num_agents[0],)),
            jax.random.choice(k2, 9, (env.num_agents[1],)),
        ]
        _state, timesteps = env.step(_state, actions)
        return _state, [t.observation for t in timesteps]

    state, _ = env.reset(key)

    state, obs = jax.lax.scan(step, state, None, 50)

    assert isinstance(state, State)
    assert isinstance(obs, list)
    assert len(obs) == 2
    assert all([isinstance(x, Observations) for x in obs])
