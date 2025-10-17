import jax.numpy as jnp

from thants.generators.colonies.mono import BasicColonyGenerator


def test_basic_colony_generator(key) -> None:
    env_dims = (5, 10)
    n_agents = 8
    n_signals = 2

    colony_generator = BasicColonyGenerator(n_agents, n_signals, (2, 2))

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
    assert colony.ants.carrying.shape == (n_agents,)
    assert colony.signals.shape == (n_signals, *env_dims)
