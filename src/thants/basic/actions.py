import chex
import jax.numpy as jnp

from .types import Actions, SignalActions


def derive_actions(actions: chex.Array) -> Actions:
    """
    Derive environment updates from integer action choices

    Maps integer to discrete actions as follows:

    - 0: Take no action
    - 1-4 Move in one of the cardinal directions
    - 5: Take a fixed amount of food
    - 6: Deposit a fixed amount of food
    - 7+: Deposit a fixed amount of sigal of type `i - 7`

    Parameters
    ----------
    actions
        Array of integer action choices

    Returns
    -------
    Actions
        Struct of actions to apply to the environment
    """
    directions = jnp.array([[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1]])

    movement_idxs = jnp.where(actions < 5, actions, 0)
    movements = directions.at[movement_idxs].get()

    take_food = jnp.where(actions == 5, 0.1, 0)
    deposit_food = jnp.where(actions == 6, 0.1, 0)

    deposit_signals = SignalActions(
        channel=jnp.maximum(actions - 7, 0),
        amount=jnp.where(actions > 6, 0.1, 0),
    )

    return Actions(
        movements=movements,
        take_food=take_food,
        deposit_food=deposit_food,
        deposit_signals=deposit_signals,
    )
