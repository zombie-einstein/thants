import chex
import jax
import jax.numpy as jnp

from thants.signals import BasicSignalPropagator
from thants.steps import deposit_signals
from thants.types import SignalActions


def test_total_signals_preserved_single(key) -> None:
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


def test_total_signals_preserved_multi(key) -> None:
    signals = jnp.zeros((3, 2, 3, 3))
    signals = signals.at[0, 0, 1, 1].set(1.0)
    signals = signals.at[1, 0, 1, 1].set(1.0)

    propagator = BasicSignalPropagator(0.0, 0.1)

    def step(s: chex.Array, _: None) -> tuple[chex.Array, chex.Array]:
        s = propagator(key, s)
        return s, s

    _, signal_ts = jax.lax.scan(step, signals, None, 20)

    assert signal_ts.shape == (20, 3, 2, 3, 3)

    total_signals = jnp.sum(signal_ts, axis=(3, 4))

    assert jnp.allclose(total_signals[:, 0, 0], 1.0)
    assert jnp.allclose(total_signals[:, 0, 1], 0.0)
    assert jnp.allclose(total_signals[:, 1, 0], 1.0)
    assert jnp.allclose(total_signals[:, 1, 1], 0.0)
    assert jnp.allclose(total_signals[:, 2, :], 0.0)


def test_deposit_signals() -> None:
    signals = jnp.zeros((1, 2, 3, 3), dtype=float)
    pos = jnp.array([[1, 1]])
    deposits = SignalActions(
        channel=jnp.array([0]),
        amount=jnp.array([0.5]),
    )
    colony_idxs = jnp.array([0])
    expected = signals.at[0, 0, 1, 1].set(0.5)
    new_signals = deposit_signals(signals, pos, colony_idxs, deposits)
    assert jnp.allclose(new_signals, expected)
