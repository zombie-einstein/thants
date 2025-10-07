import chex
import jax
import jax.numpy as jnp

from thants.basic.types import State
from thants.common.types import Observations


def observations_from_state(state: State) -> Observations:
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
    dims = state.colony.nest.shape
    idxs = jnp.indices((3, 3))
    idxs = idxs.swapaxes(0, 2).reshape(9, 2) - 1
    dims_arr = jnp.array([dims])

    view_idxs = jax.vmap(lambda x: (idxs + x) % dims_arr)(state.colony.ants.pos)

    def get_view(arr: chex.Array, x: chex.Array) -> chex.Array:
        return arr.at[x[:, 0], x[:, 1]].get()

    def get_signals(arr: chex.Array, x: chex.Array) -> chex.Array:
        return arr.at[:, x[:, 0], x[:, 1]].get()

    occupation = jnp.zeros(dims, dtype=float)
    occupation = occupation.at[
        state.colony.ants.pos[:, 0], state.colony.ants.pos[:, 1]
    ].set(1.0)

    ants = jax.vmap(get_view, in_axes=(None, 0))(occupation, view_idxs)
    food = jax.vmap(get_view, in_axes=(None, 0))(state.food, view_idxs)
    signals = jax.vmap(get_signals, in_axes=(None, 0))(state.colony.signals, view_idxs)
    nest = jax.vmap(get_view, in_axes=(None, 0))(state.colony.nest, view_idxs).astype(
        float
    )
    terrain = jax.vmap(get_view, in_axes=(None, 0))(state.terrain, view_idxs).astype(
        float
    )

    return Observations(
        ants=ants,
        food=food,
        signals=signals,
        nest=nest,
        carrying=state.colony.ants.carrying,
        terrain=terrain,
    )
