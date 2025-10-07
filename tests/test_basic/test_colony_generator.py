import jax.numpy as jnp

from thants.basic.colony_generator import BasicColonyGenerator


def test_basic_colony_generator(key) -> None:
    env_dims = (5, 10)
    n_agents = 8

    colony_generator = BasicColonyGenerator(n_agents, 2, (2, 2))

    colony = colony_generator(env_dims, key)

    assert colony.ants.pos.shape == (n_agents, 2)
    assert jnp.all(
        jnp.logical_and(0 <= colony.ants.pos[:, 0], colony.ants.pos[:, 0] < env_dims[0])
    )
    assert jnp.all(
        jnp.logical_and(0 <= colony.ants.pos[:, 1], colony.ants.pos[:, 1] < env_dims[1])
    )

    assert colony.nest.shape == env_dims
    assert jnp.isclose(jnp.sum(colony.nest), 4)
