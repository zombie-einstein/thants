import chex
import jax
import jax.numpy as jnp
import pytest

from thants.steps import (
    clear_nest,
    merge_colonies,
    update_food,
    update_positions,
)
from thants.types import Ants, Colonies, Colony


@pytest.mark.parametrize(
    (
        "food_pos, food_amount, agent_pos, take, deposit, "
        "carrying, expected_food, expected_carrying"
    ),
    [
        # Attempt to take 0 food
        ([(0, 0)], [0.0], [(0, 0)], [1.0], [0.0], [0.0], [0.0, 0.0, 0.0, 0.0], [0.0]),
        # Take available amount
        ([(0, 0)], [0.1], [(0, 0)], [1.0], [0.0], [0.0], [0.0, 0.0, 0.0, 0.0], [0.1]),
        # Take wanted amount
        ([(0, 0)], [1.0], [(0, 0)], [0.1], [0.0], [0.0], [0.9, 0.0, 0.0, 0.0], [0.1]),
        # Deposit fixed
        ([(1, 1)], [0.0], [(1, 1)], [0.0], [0.5], [1.0], [0.0, 0.0, 0.0, 0.5], [0.5]),
        # Deposit carrying
        ([(1, 1)], [0.0], [(1, 1)], [0.0], [1.0], [0.5], [0.0, 0.0, 0.0, 0.5], [0.0]),
    ],
)
def test_update_food(
    food_pos: list[tuple[int, int]],
    food_amount: list[float],
    agent_pos: list[tuple[int, int]],
    take: list[float],
    deposit: list[float],
    carrying: list[float],
    expected_food: list[float],
    expected_carrying: list[float],
) -> None:
    food_pos = jnp.array(food_pos)
    food_amount = jnp.array(food_amount)
    food = jnp.zeros((2, 2), dtype=float)
    food = food.at[food_pos[:, 0], food_pos[:, 1]].set(food_amount)

    agent_pos = jnp.array(agent_pos)
    take = jnp.array(take)
    deposit = jnp.array(deposit)
    carrying = jnp.array(carrying)

    new_food, new_carrying = update_food(food, agent_pos, take, deposit, carrying, 1.0)

    expected_food = jnp.array(expected_food).reshape(2, 2)
    expected_carrying = jnp.array(expected_carrying)

    assert jnp.allclose(new_food, expected_food)
    assert jnp.allclose(new_carrying, expected_carrying)


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


def test_merge_colonies():
    dims = (2, 2)

    colony_a = Colony(
        ants=Ants(
            pos=jnp.array([[0, 0], [0, 1]]),
            carrying=jnp.zeros((2,)),
            health=jnp.ones((2,)),
        ),
        nest=jnp.zeros(dims, dtype=bool).at[0, 0].set(True),
        signals=jnp.zeros((2, *dims)),
    )
    colony_b = Colony(
        ants=Ants(
            pos=jnp.array([[1, 1]]),
            carrying=jnp.zeros((1,)),
            health=jnp.ones((1,)),
        ),
        nest=jnp.zeros(dims, dtype=bool).at[1, 1].set(True),
        signals=jnp.zeros((2, *dims)),
    )

    colonies = merge_colonies([colony_a, colony_b])

    assert isinstance(colonies, Colonies)

    expected_pos = jnp.array([[0, 0], [0, 1], [1, 1]])
    assert jnp.array_equal(colonies.ants.pos, expected_pos)
    assert colonies.ants.carrying.shape == (3,)
    assert colonies.ants.health.shape == (3,)
    expected_nests = jnp.array([[1, 0], [0, 2]])
    assert jnp.array_equal(colonies.nests, expected_nests)
    assert colonies.signals.shape == (2, 2, *dims)
    expected_idxs = jnp.array([0, 0, 1])
    assert jnp.array_equal(colonies.colony_idx, expected_idxs)


def test_clear_food_from_nest() -> None:
    dims = (3, 1)

    nests = jnp.array([[1], [0], [2]])
    food = jnp.ones(dims)

    new_food = clear_nest(nests, jnp.ones(dims, dtype=bool), food)
    expected = jnp.array([[0.0], [1.0], [0.0]])

    assert jnp.allclose(new_food, expected)


def test_clear_food_terrain() -> None:
    dims = (3, 1)

    nests = jnp.zeros(dims)
    food = jnp.ones(dims)
    terrain = jnp.array([[True], [False], [True]])

    new_food = clear_nest(nests, terrain, food)
    expected = jnp.array([[1.0], [0.0], [1.0]])

    assert jnp.allclose(new_food, expected)
