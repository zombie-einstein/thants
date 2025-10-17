import jax.numpy as jnp

from thants.rewards import DeliveredFoodRewards, delivered_food
from thants.steps import merge_colonies
from thants.types import Ants, Colony, State


def test_delivered_food_rewards() -> None:
    dims = (3, 1)
    nest = jnp.ones(dims, dtype=bool).at[0, 0].set(False)
    pos = jnp.array([[0, 0], [1, 0], [2, 0]])
    carrying_before = jnp.array([1.0, 1.0, 1.0])
    carrying_after = jnp.array([0.5, 0.5, 1.0])

    rewards = delivered_food(nest, pos, carrying_before, carrying_after)

    expected = jnp.array([0.0, 0.5, 0.0])

    assert jnp.allclose(rewards, expected)


def test_food_reward_function(key):
    dims = (3, 2)

    nest_a = jnp.zeros(dims, dtype=bool).at[0, :].set(True)
    nest_b = jnp.zeros(dims, dtype=bool).at[2, :].set(True)

    pos_a = jnp.array([[0, 0], [1, 0], [2, 0]])
    pos_b = jnp.array([[0, 1], [1, 1], [2, 1]])

    carrying_before = jnp.ones((3,))
    carrying_after = 0.5 * jnp.ones((3,))

    signals = jnp.zeros((1, *dims))
    food = jnp.zeros(dims)
    terrain = jnp.ones(dims, dtype=bool)
    health = jnp.ones((3,))

    s0 = State(
        step=0,
        key=key,
        colonies=merge_colonies(
            [
                Colony(
                    ants=Ants(pos=pos_a, carrying=carrying_before, health=health),
                    nest=nest_a,
                    signals=signals,
                ),
                Colony(
                    ants=Ants(pos=pos_b, carrying=carrying_before, health=health),
                    nest=nest_b,
                    signals=signals,
                ),
            ]
        ),
        food=food,
        terrain=terrain,
    )

    s1 = State(
        step=0,
        key=key,
        colonies=merge_colonies(
            [
                Colony(
                    ants=Ants(pos=pos_a, carrying=carrying_after, health=health),
                    nest=nest_a,
                    signals=signals,
                ),
                Colony(
                    ants=Ants(pos=pos_b, carrying=carrying_after, health=health),
                    nest=nest_b,
                    signals=signals,
                ),
            ]
        ),
        food=food,
        terrain=terrain,
    )

    reward_fn = DeliveredFoodRewards()

    rewards = reward_fn([3, 3], s0, s1)

    assert isinstance(rewards, list)
    assert len(rewards) == 2

    expected_a = jnp.array([0.5, 0.0, 0.0])
    assert jnp.allclose(rewards[0], expected_a)

    expected_b = jnp.array([0.0, 0.0, 0.5])
    assert jnp.allclose(rewards[1], expected_b)
