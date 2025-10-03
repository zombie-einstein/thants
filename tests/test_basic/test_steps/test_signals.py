import jax.numpy as jnp

from thants.common.steps import deposit_signals
from thants.common.types import SignalActions


def test_deposit_signals() -> None:
    signals = jnp.zeros((2, 3, 3), dtype=float)
    pos = jnp.array([[1, 1]])
    deposits = SignalActions(
        channel=jnp.array([0]),
        amount=jnp.array([0.5]),
    )
    expected = signals.at[0, 1, 1].set(0.5)
    new_signals = deposit_signals(signals, pos, deposits)
    assert jnp.allclose(new_signals, expected)
