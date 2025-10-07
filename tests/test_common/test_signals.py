import chex
import jax
import jax.numpy as jnp

from thants.common.signals import BasicSignalPropagator
from thants.common.steps import deposit_signals
from thants.common.types import SignalActions


def test_total_signals_preserved(key) -> None:
    signals = jnp.zeros((2, 3, 3))
    signals = signals.at[0, 1, 1].set(1.0)

    propagator = BasicSignalPropagator(0.0, 0.1)

    def step(s: chex.Array, _: None) -> tuple[chex.Array, chex.Array]:
        s = propagator(key, s)
        return s, s

    _, signal_ts = jax.lax.scan(step, signals, None, 20)

    assert signal_ts.shape == (20, 2, 3, 3)

    total_signals = jnp.sum(signal_ts, axis=(2, 3))

    assert jnp.allclose(total_signals[:, 0], 1.0)
    assert jnp.allclose(total_signals[:, 1], 0.0)


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
