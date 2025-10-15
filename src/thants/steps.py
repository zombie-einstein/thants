from typing import Sequence

import chex
import jax.numpy as jnp

from thants.types import Ants, Colonies, Colony, SignalActions


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


def update_positions(
    dims: tuple[int, int], pos: chex.Array, terrain: chex.Array, updates: chex.Array
) -> chex.Array:
    """
    Update agent positions, checking for collisions

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


def clear_nest(nests: chex.Array, food: chex.Array) -> chex.Array:
    """
    Clear food deposited on each colony nest

    Parameters
    ----------
    nests
        List of nest flag arrays
    food
        Food state

    Returns
    -------
    chex.Array
        Food state
    """
    return jnp.where(nests > 0, 0.0, food)


def merge_colonies(colonies: Sequence[Colony]) -> Colonies:
    """
    Merge a list of colonies into a single state

    Parameters
    ----------
    colonies
        List of individual colonies

    Returns
    -------
    Colonies
        Joint state of the colonies, with added index appointing
        each ant to a colony
    """
    ant_pos = jnp.concatenate([c.ants.pos for c in colonies], axis=0)
    ant_carrying = jnp.concatenate([c.ants.carrying for c in colonies], axis=0)
    ant_health = jnp.concatenate([c.ants.health for c in colonies], axis=0)
    colony_idx = jnp.concatenate(
        [
            jnp.full(colony.ants.carrying.shape, i, dtype=int)
            for i, colony in enumerate(colonies)
        ]
    )
    signals = jnp.stack([c.signals for c in colonies], axis=0)
    nests = jnp.stack(
        [colony.nest.astype(int) * (i + 1) for i, colony in enumerate(colonies)], axis=0
    )
    nests = jnp.max(nests, axis=0)
    return Colonies(
        ants=Ants(
            pos=ant_pos,
            carrying=ant_carrying,
            health=ant_health,
        ),
        colony_idx=colony_idx,
        signals=signals,
        nests=nests,
    )


def deposit_signals(
    signals: chex.Array,
    pos: chex.Array,
    colony_idx: chex.Array,
    deposits: SignalActions,
) -> chex.Array:
    """
    Deposit signals for the relevant colony and channel

    Parameters
    ----------
    signals
        Current signal states
    pos
        Ant positions
    colony_idx
        Colony index of each ant
    deposits
        Signal deposit amounts

    Returns
    -------
    chex.Array
        Update signal states
    """
    return signals.at[colony_idx, deposits.channel, pos[:, 0], pos[:, 1]].add(
        deposits.amount
    )
