"""
Environment update steps
"""
import chex
import jax.numpy as jnp

from .types import SignalActions


def update_positions(
    dims: tuple[int, int], pos: chex.Array, updates: chex.Array
) -> chex.Array:
    """
    Update agent positions

    Parameters
    ----------
    dims
        Environment dimensions
    pos
        Ant positions
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
    new_pos = jnp.where(move_occupied[:, jnp.newaxis] > 0, pos, new_pos)
    return new_pos


def update_food(
    food: chex.Array,
    pos: chex.Array,
    take: chex.Array,
    deposit: chex.Array,
    carrying: chex.Array,
    capacity: float,
) -> tuple[chex.Array, chex.Array]:
    """
    Update food piles due to ant actions

    Parameters
    ----------
    food
        Food deposit state array
    pos
        Ant positions
    take
        Food pick-up action amounts for individual ants
    deposit
        Food pick-up deposit amounts for individual ants
    carrying
        Current ant carrying amounts
    capacity
        Ant carrying capacity

    Returns
    -------
    tuple[chex.Array, chex.Array]
        Tuple containing

        - Update food deposit state
        - Updated ant carrying amounts
    """
    available_food = food.at[pos[:, 0], pos[:, 1]].get()
    available_capacity = capacity - carrying
    taken_food = jnp.minimum(jnp.minimum(available_food, take), available_capacity)
    deposited = jnp.minimum(deposit, carrying)
    new_food = food.at[pos[:, 0], pos[:, 1]].add(deposited)
    new_food = new_food.at[pos[:, 0], pos[:, 1]].subtract(taken_food)
    new_carrying = carrying + taken_food - deposited
    return new_food, new_carrying


def deposit_signals(
    signals: chex.Array, pos: chex.Array, deposits: SignalActions
) -> chex.Array:
    """
    Deposit signals at ant locations

    Parameters
    ----------
    signals
        Signal state
    pos
        Ant positions
    deposits
        Amount to deposit by each ant

    Returns
    -------
    chex.Array
        Updated signal state
    """
    new_signals = signals.at[deposits.channel, pos[:, 0], pos[:, 1]].add(
        deposits.amount
    )
    return new_signals
