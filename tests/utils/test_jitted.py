import jax.numpy as jnp

from pydiffuser.utils.jitted import normalize


def test_normalize():
    arr = jnp.array([3, 4])
    normed_arr = normalize(arr=arr)
    assert normed_arr[0] == 0.6
    assert normed_arr[1] == 0.8
