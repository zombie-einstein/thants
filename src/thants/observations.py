from typing import Sequence

import chex
import jax
import jax.numpy as jnp
import numpy as np

from thants.types import Observations, State


def observations_from_state(
    colony_sizes: Sequence[int], state: State
) -> Sequence[Observations]:
    """
    Generate individual agent observations from state for each colony

    Parameters
    ----------
    colony_sizes
        List of colony sizes
    state
        Environment state

    Returns
    -------
    Sequence[Observations]
        List of structs containing observation components for each colony

        - Local neighbourhood with flags indicating neighbouring ants
        - Food amounts in the local neighbourhood
        - Local neighbourhood indicating if a cell is a nest
        - Deposited signals in the local neighbourhood
        - Food carried by each agent
        - Local environment terrain
    """
    n_colonies = len(colony_sizes)
    n_signals = state.colonies.signals.shape[1]
    dims = state.food.shape
    dims_arr = jnp.array([dims])
    idxs = jnp.indices((3, 3))
    idxs = idxs.swapaxes(0, 2).reshape(9, 2) - 1

    def get_ant_view(arr: chex.Array, i: chex.Array, x: chex.Array) -> chex.Array:
        return arr.at[i[:, jnp.newaxis], x[:, 0], x[:, 1]].get()

    def get_view(arr: chex.Array, x: chex.Array) -> chex.Array:
        return arr.at[x[:, 0], x[:, 1]].get()

    def get_signals(arr: chex.Array, i: int, x: chex.Array) -> chex.Array:
        return arr.at[i, jnp.arange(n_signals)[:, jnp.newaxis], x[:, 0], x[:, 1]].get()

    def get_nest(i: int, arr: chex.Array, x: chex.Array) -> chex.Array:
        return arr.at[x[:, 0], x[:, 1]].get() == (i + 1)

    occupation = jnp.zeros((n_colonies, *dims), dtype=float)
    occupation = occupation.at[
        state.colonies.colony_idx,
        state.colonies.ants.pos[:, 0],
        state.colonies.ants.pos[:, 1],
    ].set(1.0)

    view_idxs = jax.vmap(lambda x: (idxs + x) % dims_arr)(state.colonies.ants.pos)
    a_idxs = jax.vmap(lambda i: (i + jnp.arange(n_colonies)) % n_colonies)(
        state.colonies.colony_idx
    )
    ants = jax.vmap(get_ant_view, in_axes=(None, 0, 0))(occupation, a_idxs, view_idxs)
    food = jax.vmap(get_view, in_axes=(None, 0))(state.food, view_idxs)
    signals = jax.vmap(get_signals, in_axes=(None, 0, 0))(
        state.colonies.signals, state.colonies.colony_idx, view_idxs
    )
    nest = jax.vmap(get_nest, in_axes=(0, None, 0))(
        state.colonies.colony_idx, state.colonies.nests, view_idxs
    ).astype(float)
    terrain = jax.vmap(get_view, in_axes=(None, 0))(state.terrain, view_idxs).astype(
        float
    )

    boundaries = [0, *colony_sizes]
    boundaries = np.array(boundaries)
    boundaries = np.cumsum(boundaries)

    observations = [
        Observations(
            ants=ants[a:b],
            food=food[a:b],
            signals=signals[a:b],
            nest=nest[a:b],
            terrain=terrain[a:b],
            carrying=state.colonies.ants.carrying[a:b],
        )
        for a, b in zip(boundaries[:-1], boundaries[1:])
    ]

    return observations
