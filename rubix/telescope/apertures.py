""" This class defines the aperture mask for the observation of a galaxy.

"""
from jaxtyping import Array, Float
import jax.numpy as jnp

__all__ = ["HEXAGONAL_APERTURE", "SQUARE_APERTURE", "CIRCULAR_APERTURE"]


def HEXAGONAL_APERTURE(sbin: int) -> Float[Array, " sbin*sbin"]:
    """Creates a hexagonal aperture mask for the observation of a galaxy.

    Parameters
    ----------
    sbin : int
        The size of the spatial bin in each direction for the aperture mask.

    Returns
    -------
    jnp.ndarray
        A 1D array of the aperture mask.
    """

    sbin = int(sbin)  # Ensure that the input is an integer
    ap_region = jnp.zeros((sbin, sbin))  # Empty matrix for aperture mask
    xcentre, ycentre = sbin / 2 + 0.5, sbin / 2 + 0.5
    for x in range(1, sbin + 1):
        for y in range(1, sbin + 1):
            xx = x - xcentre
            yy = y - ycentre
            rr = (
                (2 * (sbin / 4) * (sbin * jnp.sqrt(3) / 4))
                - ((sbin / 4) * jnp.abs(yy))
                - ((sbin * jnp.sqrt(3) / 4) * jnp.abs(xx))
            )
            if (
                (rr >= 0)
                and (jnp.abs(xx) < sbin / 2)
                and (jnp.abs(yy) < (sbin * jnp.sqrt(3) / 4))
            ):
                ap_region = ap_region.at[x - 1, y - 1].set(1)
    return ap_region.flatten()


def SQUARE_APERTURE(sbin: int) -> Float[Array, " sbin*sbin"]:
    """Creates a square aperture mask for the observation of a galaxy.

    Parameters
    ----------
    sbin : int
        The size of the spatial bin in each direction for the aperture mask.

    Returns
    -------
    jnp.ndarray
        A 1D array of the aperture mask.
    """

    sbin = int(sbin)
    return jnp.ones((sbin, sbin)).flatten()


def CIRCULAR_APERTURE(sbin: int) -> Float[Array, " sbin*sbin"]:
    """Creates a circular aperture mask for the observation of a galaxy.

    Parameters
    ----------
    sbin : int
        The size of the spatial bin in each direction for the aperture mask.

    Returns
    -------
    jnp.ndarray
        A 1D array of the aperture mask.
    """
    sbin = int(sbin)
    aperture = jnp.zeros((sbin, sbin))  # Empty matrix for aperture mask
    xcentre, ycentre = sbin / 2 + 0.5, sbin / 2 + 0.5
    x = jnp.tile(jnp.arange(1, sbin + 1), (sbin, 1))
    y = jnp.tile(jnp.arange(sbin, 0, -1), (sbin, 1)).T
    xx, yy = x - xcentre, y - ycentre
    rr = jnp.sqrt(xx**2 + yy**2)
    aperture = aperture.at[rr <= sbin / 2].set(1)
    return aperture.flatten()