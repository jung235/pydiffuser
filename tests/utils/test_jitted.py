import jax.numpy as jnp

from pydiffuser.utils.jitted import normalize


def test_normalize():
    arr = jnp.array([[3, 4]])
    normed_arr = normalize(arr=arr)
    expected_arr = jnp.array([[0.6, 0.8]])
    assert jnp.all(normed_arr == expected_arr)

    arr = jnp.array([[3], [4]])
    normed_arr = normalize(arr=arr, axis=0)
    expected_arr = jnp.array([[0.6], [0.8]])
    assert jnp.all(normed_arr == expected_arr)
