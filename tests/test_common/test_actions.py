import jax.numpy as jnp

from thants.common.actions import derive_actions
from thants.common.types import Actions, SignalActions


def test_derive_actions() -> None:
    action_idxs = jnp.arange(9)

    actions = derive_actions(action_idxs, 0.1, 0.1, 0.1)

    assert isinstance(actions, Actions)

    expected_movements = jnp.array(
        [[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1], [0, 0], [0, 0], [0, 0], [0, 0]]
    )
    assert jnp.array_equal(actions.movements, expected_movements)

    expected_take_food = jnp.zeros((9,)).at[5].set(0.1)
    assert jnp.allclose(actions.take_food, expected_take_food)

    expected_deposit_food = jnp.zeros((9,)).at[6].set(0.1)
    assert jnp.allclose(actions.deposit_food, expected_deposit_food)

    i = jnp.array([7, 8])

    expected_deposit_signals = SignalActions(
        channel=jnp.zeros((9,)).at[i].set(jnp.arange(2)),
        amount=jnp.zeros((9,)).at[i].set(0.1),
    )

    assert isinstance(actions.deposit_signals, SignalActions)
    assert jnp.allclose(actions.deposit_signals.amount, expected_deposit_signals.amount)
    assert jnp.array_equal(
        actions.deposit_signals.channel, expected_deposit_signals.channel
    )
