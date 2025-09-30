import abc

import chex
import jax.numpy as jnp


class SignalPropagator(abc.ABC):
    def __init__(self, n_signal: int) -> None:
        self.n_signals = n_signal

    @abc.abstractmethod
    def __call__(self, key: chex.PRNGKey, signals: chex.Array) -> chex.Array:
        """Update signal state"""


class BasicSignalPropagator(SignalPropagator):
    def __init__(self, n_signals: int, decay_rate: float, dissipation_rate) -> None:
        self.decay_rate = decay_rate
        self.dissipation_rate = dissipation_rate
        super().__init__(n_signals)

    def __call__(self, key: chex.PRNGKey, signals: chex.Array) -> chex.Array:
        signals = jnp.maximum(signals - self.decay_rate, 0.0)
        dissipate = 0.25 * self.dissipation_rate * signals

        signals = (
            (1.0 - self.dissipation_rate) * signals
            + jnp.roll(dissipate, shift=-1, axis=1)
            + jnp.roll(dissipate, shift=1, axis=1)
            + jnp.roll(dissipate, shift=-1, axis=2)
            + jnp.roll(dissipate, shift=1, axis=2)
        )

        return signals
