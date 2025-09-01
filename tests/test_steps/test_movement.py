import chex
import jax
import jax.numpy as jnp
import pytest

from thants.steps import update_positions


@pytest.mark.parametrize(
    "pos, updates, expected",
    [
        # Move
        ([(0, 0)], [(1, 1)], [(1, 1)]),
        # Wrapped move
        ([(0, 0)], [(-1, -1)], [(1, 2)]),
        # Move around
        ([(0, 0), (1, 1)], [(1, 0), (0, 0)], [(1, 0), (1, 1)]),
        # Blocked move
        ([(0, 0), (1, 0)], [(1, 0), (0, 0)], [(0, 0), (1, 0)]),
        # Blocked move
        ([(0, 0), (1, 1)], [(1, 0), (0, -1)], [(0, 0), (1, 1)]),
    ],
)
def test_movement(
    pos: list[tuple[int, int]],
    updates: list[tuple[int, int]],
    expected: list[tuple[int, int]],
) -> None:
    pos = jnp.array(pos)
    updates = jnp.array(updates)

    new_pos = update_positions((2, 3), pos, updates)

    expected = jnp.array(expected)

    assert jnp.allclose(new_pos, expected)


def test_fuzz_movement(key: chex.PRNGKey) -> None:
    n_agents = 20
    dims = (12, 5)
    n_cells = dims[0] * dims[1]
    pos_idx = jax.random.choice(key, n_cells, shape=(n_agents,), replace=False)
    pos_x = pos_idx % dims[1]
    pos_y = (pos_idx - pos_x) // dims[1]
    start_pos = jnp.stack((pos_y, pos_x)).T
    movements = jnp.array([[-1, 0], [1, 0], [0, -1], [0, 1], [0, 0]])

    def step(carry, _):
        k, pos = carry
        k, move_key = jax.random.split(k)
        moves = jax.random.choice(move_key, movements, shape=(n_agents,))
        new_pos = update_positions(dims, pos, moves)
        return (k, new_pos), pos

    (_, final_pos), positions = jax.lax.scan(step, (key, start_pos), None, length=50)

    idxs = positions[:, :, 0] * dims[1] + positions[:, :, 1]

    def bin_count(x):
        return jnp.bincount(x, length=n_cells)

    occupation = jax.vmap(bin_count)(idxs)

    assert jnp.max(occupation) == 1
