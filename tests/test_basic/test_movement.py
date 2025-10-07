import chex
import jax
import jax.numpy as jnp
import pytest

from thants.basic.steps import update_positions


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
    dims = (2, 3)
    terrain = jnp.ones(dims, dtype=bool)

    new_pos = update_positions((2, 3), pos, terrain, updates)

    expected = jnp.array(expected)

    assert jnp.allclose(new_pos, expected)


def test_fuzz_movement(key: chex.PRNGKey) -> None:
    n_agents = 20
    dims = (12, 5)
    terrain = jnp.ones(dims, dtype=bool)
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
        new_pos = update_positions(dims, pos, terrain, moves)
        return (k, new_pos), pos

    (_, final_pos), positions = jax.lax.scan(step, (key, start_pos), None, length=50)

    idxs = positions[:, :, 0] * dims[1] + positions[:, :, 1]

    def bin_count(x):
        return jnp.bincount(x, length=n_cells)

    occupation = jax.vmap(bin_count)(idxs)

    assert jnp.max(occupation) == 1


def test_terrain_blocking() -> None:
    dims = (3, 3)
    terrain = jnp.ones(dims, dtype=bool)
    terrain = terrain.at[1, 1].set(False)
    pos = jnp.array([[0, 1]])

    # Should be blocked by terrain
    updates = jnp.array([[1, 0]])
    new_pos = update_positions(dims, pos, terrain, updates)
    assert jnp.array_equal(pos, new_pos)

    # Should not be blocked by terrain
    updates = jnp.array([[0, 1]])
    new_pos = update_positions(dims, pos, terrain, updates)
    expected_pos = jnp.array([[0, 2]])
    assert jnp.array_equal(expected_pos, new_pos)
