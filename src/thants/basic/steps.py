"""
Environment update steps
"""
import chex
import jax.numpy as jnp


def update_positions(
    dims: tuple[int, int], pos: chex.Array, terrain: chex.Array, updates: chex.Array
) -> chex.Array:
    """
    Update agent positions

    Parameters
    ----------
    dims
        Environment dimensions
    pos
        Ant positions
    terrain
        Environment terrain
    updates
        Array of position updates

    Returns
    -------
    chex.Array
        Updated agent positions
    """
    dims_arr = jnp.array([dims])
    new_pos = (pos + updates) % dims_arr
    x = jnp.concatenate([pos[:, 1], new_pos[:, 1]])
    y = jnp.concatenate([pos[:, 0], new_pos[:, 0]])
    idxs = y * dims[1] + x
    occupation = jnp.bincount(idxs, length=dims[0] * dims[1]).reshape(*dims)
    move_occupied = occupation.at[new_pos[:, 0], new_pos[:, 1]].get() - 1
    passable = terrain.at[new_pos[:, 0], new_pos[:, 1]].get()
    move_available = jnp.logical_and(move_occupied < 1, passable)
    new_pos = jnp.where(move_available[:, jnp.newaxis], new_pos, pos)
    return new_pos


def clear_nest(nest: chex.Array, food: chex.Array) -> chex.Array:
    return jnp.where(nest, 0.0, food)
