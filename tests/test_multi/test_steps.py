import jax.numpy as jnp

from thants.common.types import Ants, Colony
from thants.multi.steps import (
    clear_nest,
    merge_colonies,
    update_food,
    update_positions,
)
from thants.multi.types import Colonies


def test_colony_movement() -> None:
    dims = (5, 5)

    pos = [
        jnp.array([[0, 0], [3, 3]]),
        jnp.array([[0, 1]]),
    ]

    terrain = jnp.ones(dims, dtype=bool)

    updates = [
        jnp.array([[0, 1], [0, 1]]),
        jnp.array([[-1, 0]]),
    ]

    new_pos = update_positions(
        dims,
        pos,
        terrain,
        updates,
    )

    assert isinstance(new_pos, list)
    assert len(new_pos) == 2
    assert new_pos[0].shape == (2, 2)
    assert new_pos[1].shape == (1, 2)

    expected_0 = jnp.array([[0, 0], [3, 4]])
    assert jnp.array_equal(new_pos[0], expected_0)

    expected_1 = jnp.array([[4, 1]])
    assert jnp.array_equal(new_pos[1], expected_1)


def test_colony_food_update() -> None:
    food = jnp.zeros((2, 2)).at[1, 1].set(1.0)

    pos = [
        jnp.array([[0, 0], [1, 1]]),
        jnp.array([[0, 1]]),
    ]

    take = [jnp.array([1.0, 1.0]), jnp.array([0.0])]
    deposit = [jnp.array([0.0, 0.0]), jnp.array([1.0])]
    carrying = [jnp.array([0.0, 0.0]), jnp.array(0.5)]

    new_food, new_carrying = update_food(food, pos, take, deposit, carrying, 0.5)

    assert new_food.shape == (2, 2)
    expected_food = jnp.array([[0.0, 0.5], [0.0, 0.5]])
    assert jnp.array_equal(new_food, expected_food)

    assert isinstance(carrying, list)
    expected_carry_0 = jnp.array([0.0, 0.5])
    assert jnp.allclose(new_carrying[0], expected_carry_0)
    expected_carry_1 = jnp.array([0.0])
    assert jnp.allclose(new_carrying[1], expected_carry_1)


def test_clear_food() -> None:
    dims = (3, 1)

    nests = jnp.array([[1], [0], [2]])
    food = jnp.ones(dims)

    new_food = clear_nest(nests, food)
    expected = jnp.array([[0.0], [1.0], [0.0]])

    assert jnp.allclose(new_food, expected)


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
