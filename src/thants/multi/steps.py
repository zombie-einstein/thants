"""
Environment update steps
"""
import chex
import jax.numpy as jnp

from thants.common.steps import update_food as _update_food
from thants.common.types import Ants, Colony, SignalActions
from thants.multi.types import Colonies


def _move(
    occupation: chex.Array, pos: chex.Array, new_pos: chex.Array, terrain: chex.Array
) -> chex.Array:
    move_occupied = occupation.at[new_pos[:, 0], new_pos[:, 1]].get() - 1
    passable = terrain.at[new_pos[:, 0], new_pos[:, 1]].get()
    move_available = jnp.logical_and(move_occupied < 1, passable)
    new_pos = jnp.where(move_available[:, jnp.newaxis], new_pos, pos)
    return new_pos


def update_positions(
    dims: tuple[int, int],
    pos: list[chex.Array],
    terrain: chex.Array,
    updates: list[chex.Array],
) -> list[chex.Array]:
    """
    Update agent positions

    Parameters
    ----------
    dims
        Environment dimensions
    pos
        List of colony positions
    terrain
        Environment terrain
    updates
        Array of colony position updates

    Returns
    -------
    tuple[chex.Array, chex.Array]
        Updated agent positions
    """
    dims_arr = jnp.array([dims])

    new_pos = [(p + update) % dims_arr for p, update in zip(pos, updates)]

    x = jnp.concatenate([a[:, 1] for b in zip(pos, new_pos) for a in b])
    y = jnp.concatenate([a[:, 0] for b in zip(pos, new_pos) for a in b])

    idxs = y * dims[1] + x

    occupation = jnp.bincount(idxs, length=dims[0] * dims[1]).reshape(*dims)

    new_pos = [_move(occupation, p, np, terrain) for p, np in zip(pos, new_pos)]

    return new_pos


def update_food(
    food: chex.Array,
    pos: list[chex.Array],
    take: list[chex.Array],
    deposit: list[chex.Array],
    carrying: list[chex.Array],
    capacity: float,
) -> tuple[chex.Array, list[chex.Array]]:
    """
    Update food deposits due to ant actions

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
    tuple[chex.Array, list[chex.Array]]
        Tuple containing

        - Update food deposit state
        - List of updated ant carrying amounts for each colony
    """

    updates = [
        _update_food(food, _pos, _take, _deposit, _carrying, capacity)
        for _pos, _take, _deposit, _carrying in zip(pos, take, deposit, carrying)
    ]

    d_food = jnp.stack([x[0] - food for x in updates], axis=0)
    d_food = jnp.sum(d_food, axis=0)
    new_food = food + d_food
    new_carrying = [x[1] for x in updates]

    return new_food, new_carrying


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
    nests = jnp.any(nests, axis=0)
    return jnp.where(nests, 0.0, food)


def merge_colonies(colonies: list[Colony]) -> Colonies:
    ant_pos = jnp.concatenate([c.ants.pos for c in colonies], axis=0)
    ant_carrying = jnp.concatenate([c.ants.carrying for c in colonies], axis=0)
    ant_health = jnp.concatenate([c.ants.health for c in colonies], axis=0)
    colony_idx = jnp.concatenate(
        [jnp.full(c.ants.carrying.shape, i, dtype=int) for i, c in enumerate(colonies)]
    )
    signals = jnp.stack([c.signals for c in colonies], axis=0)
    nests = jnp.stack([c.nest for c in colonies], axis=0)

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
    return signals.at[colony_idx, deposits.channel, pos[:, 0], pos[:, 1]].add(
        deposits.amount
    )
