import chex
import jax
import jax.numpy as jnp

from .types import Observations, State


def observations_from_state(dims: tuple[int, int], state: State) -> Observations:
    idxs = jnp.indices((3, 3))
    idxs = idxs.swapaxes(0, 2).reshape(9, 2) - 1
    dims_arr = jnp.array([dims])

    view_idxs = jax.vmap(lambda x: (idxs + x) % dims_arr)(state.ants.pos)

    def get_view(arr: chex.Array, x: chex.Array) -> chex.Array:
        return arr.at[x[:, 0], x[:, 1]].get()

    occupation = jnp.zeros(dims, dtype=float)
    occupation = occupation.at[state.ants.pos[:, 0], state.ants.pos[:, 1]].set(1.0)

    ants = jax.vmap(get_view, in_axes=(None, 0))(occupation, view_idxs)
    food = jax.vmap(get_view, in_axes=(None, 0))(state.food, view_idxs)
    signals = jax.vmap(get_view, in_axes=(None, 0))(state.signals, view_idxs)
    nest = jax.vmap(get_view, in_axes=(None, 0))(state.nest, view_idxs)

    return Observations(
        ants=ants, food=food, signals=signals, nest=nest, carrying=state.ants.carrying
    )
