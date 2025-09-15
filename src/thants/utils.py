import chex
import jax.numpy as jnp


def get_rectangular_indices(rec_dims: tuple[int, int]) -> chex.Array:
    n_idxs = rec_dims[0] * rec_dims[1]
    idxs = jnp.indices(rec_dims).reshape(2, n_idxs).T
    return idxs
