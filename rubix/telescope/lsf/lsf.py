"""
Mainly reimplmented from SimSpin:
https://github.com/kateharborne/SimSpin/blob/4e8f0af0ebc0e43cc31729978deb3a554e039f6b/R/utilities.R#L570
"""

import jax.numpy as jnp
from jax.scipy.signal import convolve
from jax import vmap
from jaxtyping import Float, Array


def gaussian1d(x: Float[Array, " n_x"], sigma: float) -> Float[Array, " n_x"]:
    res = jnp.exp(-0.5 * (x**2) / sigma**2)
    # return jnp.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * jnp.sqrt(2 * jnp.pi))
    return res / jnp.sum(res)


def _convolve_kernel(spec, kernel, mode="full"):
    return convolve(spec, kernel, mode=mode)


def _get_kernel(sigma: float, wave_res: float, factor: int = 12):
    x = jnp.arange(-factor * wave_res, factor * wave_res + wave_res, step=wave_res)

    kernel = gaussian1d(x, sigma)
    return kernel


def apply_lsf(
    spectra: Float[Array, "n_stars wave_bins"],
    lsf_sigma: float,
    wave_resolution: float,
    extend_factor: int = 12,
) -> Float[Array, "n_stars wave_bins"]:
    kernel = _get_kernel(lsf_sigma, wave_resolution, factor=extend_factor)

    # Vmap the convolution across all stars
    convolved = vmap(_convolve_kernel, in_axes=(0, None))(spectra, kernel)

    end = spectra.shape[1] + kernel.shape[0] - 1 - extend_factor
    return convolved[:, extend_factor:end]
