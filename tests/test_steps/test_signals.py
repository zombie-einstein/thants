import jax.numpy as jnp

from thants.steps import deposit_signals, update_signals


def test_deposit_signals() -> None:
    signals = jnp.zeros((3, 3), dtype=float)
    pos = jnp.array([[1, 1]])
    deposits = jnp.array([0.5])
    expected = signals.at[1, 1].set(0.5)
    new_signals = deposit_signals(signals, pos, deposits)
    assert jnp.allclose(new_signals, expected)


def test_dissipate_signals() -> None:
    signals = jnp.zeros((3, 3), dtype=float)
    signals = signals.at[1, 1].set(0.2)

    new_signals = update_signals(signals, 0.0, 0.1)
    assert jnp.isclose(new_signals[1, 1], 0.1)

    new_signals = update_signals(new_signals, 0.0, 0.1)
    assert jnp.allclose(new_signals, 0.0)

    new_signals = update_signals(new_signals, 0.0, 0.1)
    assert jnp.allclose(new_signals, 0.0)
