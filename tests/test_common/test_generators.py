import jax.numpy as jnp

from thants.common.generators.food import BasicFoodGenerator


def test_food_generator(key) -> None:
    env_dims = (5, 10)

    food_generator = BasicFoodGenerator((2, 2), 10, 0.5)
    food = food_generator.init(env_dims, key)

    assert food.shape == env_dims
    assert jnp.isclose(jnp.sum(food), 2.0)
