"""Helper functions for alignment tasks.


Some of the helper function in this module were taken from Kate Harborne's
SimSpin code.
"""

import jax.numpy as jnp
from jaxtyping import Float, Array
from typing import Tuple


def center_particles(
    stellar_coordinates: Float[Array, " n_stars 3"],
    stellar_velocities: Float[Array, " n_stars 3"],
    galaxy_center: Float[Array, "3"],
) -> Tuple[Float[Array, " n_stars 3"], Float[Array, " n_stars 3"]]:
    """Center the stellar particles around the galaxy center.

    Parameters
    ----------
    stellar_coordinates : jnp.ndarray
        The coordinates of the stellar particles.
    stellar_velocities : jnp.ndarray
        The velocities of the stellar particles.
    galaxy_center : jnp.ndarray
        The center of the galaxy.

    Returns
    -------
    jnp.ndarray
        The new coordinates of the stellar particles.
    jnp.ndarray
        The new velocities of the stellar particles.
    """
    # Check if Center is within bounds
    check_bounds = (
        (galaxy_center[0] >= jnp.min(stellar_coordinates[:, 0]))
        & (galaxy_center[0] <= jnp.max(stellar_coordinates[:, 0]))
        & (galaxy_center[1] >= jnp.min(stellar_coordinates[:, 1]))
        & (galaxy_center[1] <= jnp.max(stellar_coordinates[:, 1]))
        & (galaxy_center[2] >= jnp.min(stellar_coordinates[:, 2]))
        & (galaxy_center[2] <= jnp.max(stellar_coordinates[:, 2]))
    )

    if not check_bounds:
        raise ValueError("Center is not within the bounds of the galaxy")

    # Calculate Central Velocity from median velocities within 10kpc of center
    mask = jnp.linalg.norm(stellar_coordinates - galaxy_center, axis=1) < 10
    # TODO this should be a median
    central_velocity = jnp.median(stellar_velocities[mask], axis=0)

    new_stellar_coordinates = stellar_coordinates - galaxy_center
    new_stellar_velocities = stellar_velocities - central_velocity
    return new_stellar_coordinates, new_stellar_velocities
