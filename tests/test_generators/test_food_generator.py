import chex
import jax
import jax.numpy as jnp

from thants.generators.food import BasicFoodGenerator


def test_food_generator(key: chex.PRNGKey) -> None:
    env_dims = (5, 10)

    k1, k2 = jax.random.split(key)

    food_generator = BasicFoodGenerator((2, 2), 10, 0.5)
    food = food_generator.init(env_dims, k1)

    assert food.shape == env_dims
    assert jnp.isclose(jnp.sum(food), 2.0)

    food = food_generator.update(k2, 4, food)

    assert food.shape == env_dims
    assert jnp.isclose(jnp.sum(food), 2.0)

    food = food_generator.update(k2, 9, food)

    assert food.shape == env_dims
    assert jnp.isclose(jnp.sum(food), 4.0)
