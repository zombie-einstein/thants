import jax.numpy as jnp

from thants.generator import BasicGenerator


def test_basic_generator(key):
    env_dims = (5, 10)
    n_agents = 8

    generator = BasicGenerator(env_dims, n_agents, (2, 2), (2, 2), 10, 0.5)

    ants, nest, food = generator.init(key)

    assert ants.pos.shape == (n_agents, 2)
    assert jnp.all(jnp.logical_and(0 <= ants.pos[:, 0], ants.pos[:, 0] < env_dims[0]))
    assert jnp.all(jnp.logical_and(0 <= ants.pos[:, 1], ants.pos[:, 1] < env_dims[1]))

    assert nest.shape == env_dims
    assert jnp.isclose(jnp.sum(nest), 4)

    assert food.shape == env_dims
    assert jnp.isclose(jnp.sum(food), 2.0)
