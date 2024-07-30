from functools import partial

import jax
import jax.numpy as jnp
import pytest
from jax.random import PRNGKey
from numpy.random import rand, randint, uniform
from scipy.stats import expon, norm, pareto

import pydiffuser as pyd
from pydiffuser.utils.jitted import (
    get_noise,
    normalize,
    polar_to_cartesian,
    spherical_to_cartesian,
)


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
    "r, phi",
    [
        (1, [jnp.pi / 6, jnp.pi / 4, jnp.pi / 3, jnp.pi / 2, jnp.pi]),
        (2, [jnp.pi / 6, jnp.pi / 4, jnp.pi / 3, jnp.pi / 2, jnp.pi]),
    ],
)
def test_polar_to_cartesian(r, phi):
    phi = jnp.array(phi)
    x1, x2 = polar_to_cartesian(r, phi)
    assert jnp.all(x1 == r * jnp.cos(phi))
    assert jnp.all(x2 == r * jnp.sin(phi))


def test_spherical_to_cartesian():
    r = 1
    theta = rand(50) * jnp.pi
    phi = rand(50) * 2 * jnp.pi
    x1, x2, x3 = spherical_to_cartesian(
        r=r, theta=theta.reshape(5, 10), phi=phi.reshape(5, 10)
    )
    assert x1.shape == (5, 10)
    assert x2.shape == (5, 10)
    assert x3.shape == (5, 10)


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
