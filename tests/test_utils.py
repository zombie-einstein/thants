import jax.numpy as jnp

from thants.utils import get_rectangular_indices


def test_rectangular_indices() -> None:
    idxs = get_rectangular_indices((2, 3))
    expected = jnp.array([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]])
    assert jnp.array_equal(idxs, expected)
