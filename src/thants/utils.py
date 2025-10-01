import chex
import jax.numpy as jnp


def get_rectangular_indices(rec_dims: tuple[int, int]) -> chex.Array:
    """
    Get cell indices for a rectangle

    Parameters
    ----------
    rec_dims
        Dimensions of the rectangle

    Returns
    -------
    chex.Array
        Array of indices with shape [n, 2]
    """
    n_idxs = rec_dims[0] * rec_dims[1]
    idxs = jnp.indices(rec_dims).reshape(2, n_idxs).T
    return idxs
