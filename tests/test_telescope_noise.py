import pytest
import jax.numpy as jnp
import jax.random as jrandom
from rubix.telescope.noise.noise import calculate_noise_cube


def check_noise_variation(noise_cube):
    """Check that the noise cube does not have the same noise in each telescope bin."""
    flat_noise = noise_cube.reshape(-1, noise_cube.shape[-1])
    variances = jnp.var(flat_noise, axis=0)
    return jnp.all(variances > 0)


def test_calculate_noise_cube_standard_case():
    key = jrandom.PRNGKey(0)
    cube = jrandom.uniform(key, shape=(5, 5, 10))
    S2N = jrandom.uniform(key, shape=(5, 5))

    noise_cube = calculate_noise_cube(cube, S2N)

    assert noise_cube.shape == cube.shape
    assert not jnp.isnan(noise_cube).any()
    assert check_noise_variation(noise_cube)


def test_calculate_noise_cube_with_zeros():
    key = jrandom.PRNGKey(0)
    cube = jnp.zeros((5, 5, 10))
    S2N = jrandom.uniform(key, shape=(5, 5))

    noise_cube = calculate_noise_cube(cube, S2N)

    assert noise_cube.shape == cube.shape
    assert jnp.all(noise_cube == 0)
    # assert check_noise_variation(
    #     noise_cube
    # )


def test_calculate_noise_cube_infinite_S2N():
    key = jrandom.PRNGKey(0)
    cube = jrandom.uniform(key, shape=(5, 5, 10))
    S2N = jnp.array([[float("inf")] * 5] * 5)

    noise_cube = calculate_noise_cube(cube, S2N)

    assert noise_cube.shape == cube.shape
    assert not jnp.isinf(noise_cube).any()
    # assert check_noise_variation(noise_cube) # This test will fail, since all values are zero


def test_calculate_noise_cube_negative_S2N():
    key = jrandom.PRNGKey(0)
    cube = jrandom.uniform(key, shape=(5, 5, 10))
    S2N = -jrandom.uniform(key, shape=(5, 5))

    noise_cube = calculate_noise_cube(cube, S2N)

    assert noise_cube.shape == cube.shape
    assert not jnp.isnan(noise_cube).any()
    assert check_noise_variation(noise_cube)
