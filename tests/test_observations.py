import chex
import jax.numpy as jnp

from thants.observations import observations_from_state
from thants.types import Ants, Observations, State


def test_observations_from_state(key: chex.PRNGKey) -> None:
    dims = (3, 2)

    ant_pos = jnp.array([[0, 0], [1, 1]])
    ant_health = jnp.zeros(2)
    ant_carry = jnp.zeros(2)

    food = jnp.zeros(dims)
    signals = jnp.zeros((2, dims[0], dims[1]))
    nest = jnp.zeros(dims)

    state = State(
        step=0,
        key=key,
        ants=Ants(
            pos=ant_pos,
            health=ant_health,
            carrying=ant_carry,
        ),
        food=food,
        signals=signals,
        nest=nest,
    )

    observations = observations_from_state(state)

    assert isinstance(observations, Observations)

    assert observations.ants.shape == (2, 9)
    assert observations.food.shape == (2, 9)
    assert observations.signals.shape == (2, 2, 9)
    assert observations.nest.shape == (2, 9)
    assert observations.carrying.shape == (2,)
