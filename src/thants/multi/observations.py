import chex
import jax
import jax.numpy as jnp

from thants.common.types import Colony, Observations
from thants.multi.types import State


def observations_from_state(state: State) -> list[Observations]:
    """
    Generate individual agent observations from state

    Parameters
    ----------
    state
        Environment state

    Returns
    -------
    Observations
        Struct containing observation components

        - Local neighbourhood with flags indicating neighbouring ants
        - Food amounts in the local neighbourhood
        - Local neighbourhood indicating if a cell is a nest
        - Deposited signals in the local neighbourhood
    """
    n_colonies = len(state.colonies)
    dims = state.food.shape
    dims_arr = jnp.array([dims])
    idxs = jnp.indices((3, 3))
    idxs = idxs.swapaxes(0, 2).reshape(9, 2) - 1

    def get_ant_view(arr: chex.Array, i: chex.Array, x: chex.Array) -> chex.Array:
        return arr.at[i[:, jnp.newaxis], x[:, 0], x[:, 1]].get()

    def get_view(arr: chex.Array, x: chex.Array) -> chex.Array:
        return arr.at[x[:, 0], x[:, 1]].get()

    def get_signals(arr: chex.Array, x: chex.Array) -> chex.Array:
        return arr.at[:, x[:, 0], x[:, 1]].get()

    occupation = jnp.zeros(dims, dtype=float)
    occupations = [
        occupation.at[c.ants.pos[:, 0], c.ants.pos[:, 1]].set(1.0)
        for c in state.colonies
    ]
    occupation = jnp.stack(occupations, axis=0)

    def get_observation(i, colony: Colony) -> Observations:
        view_idxs = jax.vmap(lambda x: (idxs + x) % dims_arr)(colony.ants.pos)
        a_idxs = (i + jnp.arange(n_colonies)) % n_colonies
        ants = jax.vmap(get_ant_view, in_axes=(None, None, 0))(
            occupation, a_idxs, view_idxs
        )
        food = jax.vmap(get_view, in_axes=(None, 0))(state.food, view_idxs)
        signals = jax.vmap(get_signals, in_axes=(None, 0))(colony.signals, view_idxs)
        nest = jax.vmap(get_view, in_axes=(None, 0))(colony.nest, view_idxs).astype(
            float
        )
        terrain = jax.vmap(get_view, in_axes=(None, 0))(
            state.terrain, view_idxs
        ).astype(float)
        return Observations(
            ants=ants,
            food=food,
            signals=signals,
            nest=nest,
            carrying=colony.ants.carrying,
            terrain=terrain,
        )

    return [get_observation(i, colony) for i, colony in enumerate(state.colonies)]
