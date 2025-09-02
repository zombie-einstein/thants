import chex
import jax.numpy as jnp

from .types import Actions


def derive_actions(actions: chex.Array) -> Actions:
    directions = jnp.array([[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1]])

    movement_idxs = jnp.where(actions < 5, actions, 0)
    movements = directions.at[movement_idxs].get()

    take_food = jnp.where(actions == 5, 0.1, 0)
    deposit_food = jnp.where(actions == 6, 0.1, 0)
    deposit_signals = jnp.where(actions == 7, 0.1, 0)

    return Actions(
        movements=movements,
        take_food=take_food,
        deposit_food=deposit_food,
        deposit_signals=deposit_signals,
    )
