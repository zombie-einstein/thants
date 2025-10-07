import jax.numpy as jnp

from thants.common.types import Colony
from thants.multi.colonies_generator import BasicColoniesGenerator


def test_colony_generator(key):
    dims = (50, 100)
    generator = BasicColoniesGenerator(25, 2, (5, 5))
    colonies = generator(dims, key)
    assert isinstance(colonies, list)
    assert len(colonies) == 2
    assert all([isinstance(c, Colony) for c in colonies])
    assert all([c.ants.pos.shape == (25, 2) for c in colonies])

    occupation = jnp.zeros(dims, dtype=int)
    pos_0 = colonies[0].ants.pos
    occupation = occupation.at[pos_0[:, 0], pos_0[:, 1]].add(1)
    pos_1 = colonies[1].ants.pos
    occupation = occupation.at[pos_1[:, 0], pos_1[:, 1]].add(1)

    assert jnp.sum(occupation) == 50
    assert jnp.max(occupation) == 1
