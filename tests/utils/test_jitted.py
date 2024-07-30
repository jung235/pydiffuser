from functools import partial

import jax
import jax.numpy as jnp
import pytest
from jax.random import PRNGKey
from numpy.random import rand, randint, uniform
from scipy.stats import expon, norm, pareto

import pydiffuser as pyd
from pydiffuser.utils.jitted import get_noise, normalize


def test_normalize():
    arr = jnp.array([[3, 4]])
    normed_arr = normalize(arr=arr)
    expected_arr = jnp.array([[0.6, 0.8]])
    assert jnp.all(normed_arr == expected_arr)

    arr = jnp.array([[3], [4]])
    normed_arr = normalize(arr=arr, axis=0)
    expected_arr = jnp.array([[0.6], [0.8]])
    assert jnp.all(normed_arr == expected_arr)


@pytest.mark.parametrize(
    "generator",
    [
        rand,
        partial(rand),
        partial(randint, 0, 2),
        partial(uniform, -1, 1),
        partial(expon.rvs, scale=10),
        norm.rvs,
        partial(pareto.rvs, b=1.5),
        partial(jax.random.uniform, key=PRNGKey(42)),
        partial(jax.random.normal, key=PRNGKey(42)),
        partial(jax.random.pareto, key=PRNGKey(42), b=1.5),
        partial(jax.random.exponential, key=PRNGKey(42)),
        partial(pyd.random.exponential, key=PRNGKey(42), scale=10),
    ],
)
def test_get_noise(generator):
    p = get_noise(generator=generator, size=500, shape=(5, 50, 2))
    assert p.shape == (5, 50, 2)
