import jax.numpy as jnp

from thants.actions import derive_actions
from thants.types import Actions


def test_derive_actions() -> None:
    action_idxs = jnp.arange(8)

    actions = derive_actions(action_idxs)

    assert isinstance(actions, Actions)

    expected_movements = jnp.array(
        [[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1], [0, 0], [0, 0], [0, 0]]
    )
    assert jnp.array_equal(actions.movements, expected_movements)

    expected_take_food = jnp.zeros((8,)).at[5].set(0.1)
    assert jnp.allclose(actions.take_food, expected_take_food)

    expected_deposit_food = jnp.zeros((8,)).at[6].set(0.1)
    assert jnp.allclose(actions.deposit_food, expected_deposit_food)

    expected_deposit_signals = jnp.zeros((8,)).at[7].set(0.1)
    assert jnp.allclose(actions.deposit_signals, expected_deposit_signals)
