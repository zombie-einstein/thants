import jax.numpy as jnp

from thants.generators.colonies.multi import DualBasicColoniesGenerator
from thants.types import Colony


def test_colony_generator(key) -> None:
    dims = (25, 50)
    n_agents = (25, 16)
    generator = DualBasicColoniesGenerator(n_agents, 2, (5, 5))
    colonies = generator(dims, key)
    assert isinstance(colonies, list)
    assert len(colonies) == 2
    assert all([isinstance(c, Colony) for c in colonies])
    assert all(
        [c.ants.pos.shape == (n, 2) for c, n in zip(colonies, generator.n_agents)]
    )

    occupation = jnp.zeros(dims, dtype=int)

    for n, colony in zip(generator.n_agents, colonies):
        pos = colony.ants.pos
        assert pos.shape == (n, 2)
        assert jnp.sum(colony.nest) == 25
        occupation = occupation.at[pos[:, 0], pos[:, 1]].add(1)

    assert jnp.sum(occupation) == sum(n_agents)
    assert jnp.max(occupation) == 1
