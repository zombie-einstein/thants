import chex
import jax
import jax.numpy as jnp

from thants.common.signals import BasicSignalPropagator


def test_total_signals_preserved(key) -> None:
    signals = jnp.zeros((2, 3, 3))
    signals = signals.at[0, 1, 1].set(1.0)

    propagator = BasicSignalPropagator(2, 0.0, 0.1)

    def step(s: chex.Array, _: None) -> tuple[chex.Array, chex.Array]:
        s = propagator(key, s)
        return s, s

    _, signal_ts = jax.lax.scan(step, signals, None, 20)

    assert signal_ts.shape == (20, 2, 3, 3)

    total_signals = jnp.sum(signal_ts, axis=(2, 3))

    assert jnp.allclose(total_signals[:, 0], 1.0)
    assert jnp.allclose(total_signals[:, 1], 0.0)
