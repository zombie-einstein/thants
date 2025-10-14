import chex
import jax.numpy as jnp

from thants.common.types import Ants, Colony
from thants.multi.observations import observations_from_state
from thants.multi.steps import merge_colonies
from thants.multi.types import State


def test_colony_observations(key: chex.PRNGKey) -> None:
    dims = (4, 4)

    colonies = [
        Colony(
            ants=Ants(
                pos=jnp.array([[0, 1], [1, 1]]),
                carrying=jnp.zeros((2,)),
                health=jnp.ones((2,)),
            ),
            nest=jnp.zeros(dims),
            signals=jnp.zeros((2, *dims)),
        ),
        Colony(
            ants=Ants(
                pos=jnp.array([[2, 2]]), carrying=jnp.zeros((1,)), health=jnp.ones((1,))
            ),
            nest=jnp.zeros(dims),
            signals=jnp.zeros((2, *dims)),
        ),
    ]

    colonies = merge_colonies(colonies)

    state = State(
        step=0,
        key=key,
        colonies=colonies,
        food=jnp.zeros(dims),
        terrain=jnp.ones(dims, dtype=bool),
    )

    observations = observations_from_state([2, 1], state)

    assert isinstance(observations, list)
    assert len(observations) == 2

    assert observations[0].ants.shape == (2, 2, 9)
    assert observations[1].ants.shape == (1, 2, 9)

    expected_obs_0 = jnp.zeros((2, 2, 9))
    expected_obs_0 = expected_obs_0.at[0, [0, 0], [4, 5]].set(1.0)
    expected_obs_0 = expected_obs_0.at[1, [0, 0, 1], [3, 4, 8]].set(1.0)

    assert jnp.allclose(observations[0].ants, expected_obs_0)

    expected_obs_1 = jnp.zeros((1, 2, 9))
    expected_obs_1 = expected_obs_1.at[0, [0, 1], [4, 0]].set(1.0)

    assert jnp.allclose(observations[1].ants, expected_obs_1)
