import jax.numpy as jnp
import pytest

from thants.basic.steps import update_food


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
