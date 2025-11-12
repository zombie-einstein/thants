import chex
import jax.numpy as jnp
import pytest

from thants.observations import observations_from_state
from thants.steps import merge_colonies
from thants.types import Ants, Colony, State


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

    assert observations[0].signals.shape == (2, 2, 9)
    assert observations[1].signals.shape == (1, 2, 9)


@pytest.mark.parametrize(
    "view_distance, x0, x1",
    [
        (1, [8, 7, 5, 4], [0, 2, 6, 8]),
        (2, [4, 5, 7, 8], [8, 6, 2, 0]),
        (3, [0, 0, 0, 0], [4, 4, 4, 4]),
    ],
)
def test_observation_range(
    key: chex.PRNGKey,
    view_distance: int,
    x0: list[int],
    x1: list[int],
) -> None:
    dims = (3, 3)

    colonies = [
        Colony(
            ants=Ants(
                pos=jnp.array([[0, 0], [1, 1], [2, 2]]),
                carrying=jnp.zeros((3,)),
                health=jnp.ones((3,)),
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
        food=jnp.arange(9).reshape(3, 3),
        terrain=jnp.ones(dims, dtype=bool),
    )

    observations = observations_from_state([3], state, view_distance=view_distance)

    d = 2 * view_distance + 1
    n = d**2

    assert observations[0].ants.shape == (3, 1, n)
    assert observations[0].food.shape == (3, n)
    assert observations[0].nest.shape == (3, n)

    idxs = jnp.array([0, d * (d - 1), d - 1, n - 1])

    assert jnp.array_equal(observations[0].food[0, idxs], jnp.array(x0))
    assert jnp.array_equal(observations[0].food[1, idxs], jnp.array(x1))
