import abc

import chex
import jax.numpy as jnp


class SignalPropagator(abc.ABC):
    """
    Signal dynamics base-class, responsible for updating signal state
    """

    @abc.abstractmethod
    def __call__(self, key: chex.PRNGKey, signals: chex.Array) -> chex.Array:
        """
        Generate updated signal state

        Parameters
        ----------
        key
            JAX random key
        signals
            Signal state array

        Returns
        -------
        chex.Array
            New signal state
        """


class BasicSignalPropagator(SignalPropagator):
    """
    Basic signal propagator

    Propagator that applies two steps:

    - Dissipate signal by subtracting a fixed amount from all deposits
    - Propagate signals by sharing a fraction of all deposits to neighbouring cells
    """

    def __init__(self, decay_rate: float, dissipation_rate) -> None:
        """
        Initialise a basic propagator

        Parameters
        ----------
        decay_rate
            Amount removed from deposits each step
        dissipation_rate
            Fraction of deposit shared with neighbouring cells
        """
        self.decay_rate = decay_rate
        self.dissipation_rate = dissipation_rate
        super().__init__()

    def __call__(self, key: chex.PRNGKey, signals: chex.Array) -> chex.Array:
        """
        Generate updated signal state

        Parameters
        ----------
        key
            JAX random key
        signals
            Signal state array

        Returns
        -------
        chex.Array
            New signal state
        """
        signals = jnp.maximum(signals - self.decay_rate, 0.0)
        dissipate = 0.25 * self.dissipation_rate * signals

        signals = (
            (1.0 - self.dissipation_rate) * signals
            + jnp.roll(dissipate, shift=-1, axis=-1)
            + jnp.roll(dissipate, shift=1, axis=-1)
            + jnp.roll(dissipate, shift=-1, axis=-2)
            + jnp.roll(dissipate, shift=1, axis=-2)
        )

        return signals
