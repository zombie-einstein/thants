import jax.numpy as jnp

from thants.common.rewards import delivered_food


def test_delivered_food_rewards():
    dims = (3, 1)
    nest = jnp.ones(dims, dtype=bool).at[0, 0].set(False)
    pos = jnp.array([[0, 0], [1, 0], [2, 0]])
    carrying_before = jnp.array([1.0, 1.0, 1.0])
    carrying_after = jnp.array([0.5, 0.5, 1.0])

    rewards = delivered_food(nest, pos, carrying_before, carrying_after)

    expected = jnp.array([0.0, 0.5, 0.0])

    assert jnp.allclose(rewards, expected)
