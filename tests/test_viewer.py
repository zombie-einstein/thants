import chex
import jax.numpy as jnp
import pytest
from matplotlib.animation import FuncAnimation
from matplotlib.pyplot import Figure

from thants.envs.multi import Thants
from thants.generators.colonies.multi import DualBasicColoniesGenerator
from thants.generators.food import BasicFoodGenerator


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


def test_render(monkeypatch, key: chex.PRNGKey, env: Thants) -> None:
    monkeypatch.setattr(Figure, "show", lambda _: None)
    state, _ = env.reset(key)
    env.render(state)


def test_animation(key: chex.PRNGKey, env: Thants) -> None:
    state, _ = env.reset(key)

    states = [state]

    for _ in range(2):
        actions = [jnp.zeros((n,), dtype=int) for n in env.num_agents]
        state, _ = env.step(state, actions)
        states.append(state)

    animation = env.animate(states)
    assert isinstance(animation, FuncAnimation)
