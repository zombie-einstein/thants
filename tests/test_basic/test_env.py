import chex
import jax
import pytest
from jumanji.testing.env_not_smoke import (
    check_env_does_not_smoke,
    check_env_specs_does_not_smoke,
)

from thants.basic import Thants
from thants.basic.colony_generator import BasicColonyGenerator
from thants.common.generators.food import BasicFoodGenerator
from thants.common.types import Observations


@pytest.fixture
def env() -> Thants:
    dims = (50, 50)
    colony_generator = BasicColonyGenerator(100, 2, (5, 5))
    food_generator = BasicFoodGenerator(
        (2, 2),
        50,
    )
    return Thants(
        dims=dims, colony_generator=colony_generator, food_generator=food_generator
    )


def test_env_does_not_smoke(env: Thants) -> None:
    """Test that we can run an episode without any errors."""
    env.max_steps = 100

    def select_action(action_key: chex.PRNGKey, _state: Observations) -> chex.Array:
        return jax.random.choice(action_key, 9, (env.num_agents,))

    check_env_does_not_smoke(env, select_action=select_action)


def test_env_specs_do_not_smoke(env: Thants) -> None:
    """Test that we can access specs without any errors."""
    check_env_specs_does_not_smoke(env)
