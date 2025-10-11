import chex
import jax.numpy as jnp


def delivered_food(
    nest: chex.Array,
    pos: chex.Array,
    carrying_before: chex.Array,
    carrying_after: chex.Array,
) -> chex.Array:
    d_carrying = carrying_before - carrying_after
    is_nest = nest.at[pos[:, 0], pos[:, 1]].get()
    rewards = jnp.where(is_nest, d_carrying, 0.0)
    return rewards
