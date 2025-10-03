import jax.numpy as jnp

from thants.basic.generators.ants import BasicAntGenerator
from thants.common.generators.food import BasicFoodGenerator


def test_basic_ant_generator(key) -> None:
    env_dims = (5, 10)
    n_agents = 8

    ant_generator = BasicAntGenerator(n_agents, (2, 2))

    ants, nest = ant_generator(env_dims, key)

    assert ants.pos.shape == (n_agents, 2)
    assert jnp.all(jnp.logical_and(0 <= ants.pos[:, 0], ants.pos[:, 0] < env_dims[0]))
    assert jnp.all(jnp.logical_and(0 <= ants.pos[:, 1], ants.pos[:, 1] < env_dims[1]))

    assert nest.shape == env_dims
    assert jnp.isclose(jnp.sum(nest), 4)


def test_food_generator(key) -> None:
    env_dims = (5, 10)

    food_generator = BasicFoodGenerator((2, 2), 10, 0.5)
    food = food_generator.init(env_dims, key)

    assert food.shape == env_dims
    assert jnp.isclose(jnp.sum(food), 2.0)
