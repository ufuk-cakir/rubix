import jax.numpy as jnp
from jax import random as jrandom

from jaxtyping import Array, Float


def calculate_noise_cube(
    cube: Float[Array, "n_x n_y n_wave_bins"], S2N: Float[Array, "n_y n_y"]
):
    """Calculate the noise cube given the cube and the signal-to-noise ratio.


    Adapted from: https://github.com/kateharborne/SimSpin/blob/4e8f0af0ebc0e43cc31729978deb3a554e039f6b/R/utilities.R#L587

    Parameters
    ----------
    cube : jnp.array (n_x, n_y, n_wave_bins)
        The data cube.
    S2N : jnp.array (n_y, n_y)
        The signal-to-noise ratio for each spaxel.

    Returns
    -------
    jnp.array (n_x, n_y, n_wave_bins)
        The noise cube.
    """
    key = jrandom.PRNGKey(0)
    S2N = jnp.where(
        jnp.isinf(S2N), 0, S2N
    )  # removing infinite noise where particles per pixel = 0

    # Generate noise for each element in the cube based on the S/N
    noise = jrandom.normal(key, shape=cube.shape) * S2N[:, :, None]

    # Scale the noise by the cube to get S/N
    noise = cube * noise

    return noise
