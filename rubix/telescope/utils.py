import jax.numpy as jnp
from jaxtyping import Array, Float
from rubix.cosmology.base import BaseCosmology
from typing import Tuple


def calculate_spatial_bin_edges(
    fov: float, spatial_bins: float, dist_z: float, cosmology: BaseCosmology
) -> Tuple[Float[Array, " n_bins"], float]:
    """Calculate the bin edges for the spatial bins.
    jnp.array
        The bin edges for the spatial bins.
    """
    ang_size = cosmology.angular_scale(dist_z)
    aperture_size = ang_size * fov
    spatial_bin_size = aperture_size / spatial_bins
    spatial_bin_edges = jnp.arange(
        -aperture_size / 2, aperture_size / 2, spatial_bin_size
    )
    return spatial_bin_edges, spatial_bin_size


def square_spaxel_assignment(
    coords: Float[Array, " n_stars 3"], spatial_bin_edges: Float[Array, " n_bins"]
) -> Float[Array, " n_stars"]:
    """Bin the particle coordinates into a 2D image with the given bin edges for square pixels.

    This function takes the particle coordinates and bins them into a 2D image with the given bin edges.
    The binning is done by digitizing the x and y coordinates of the particles and then calculating the
    flat indices of the 2D image.

    The returned indexes are the pixel assignments of the particles. Indexing starts at 0.

    Parameters
    ----------
    coords : jnp.array (n, 3)
        The particle coordinates.

    spatial_bin_edges : jnp.array
        The bin edges for the spatial bins.

    Returns
    -------
    jnp.array
        The flat pixel assignments of the particles. Indexing starts at 0.

    """

    # Calculate assignment of of x and y coordinates to bins separately
    x_indices = (
        jnp.digitize(coords[:, 0], spatial_bin_edges) - 1
    )  # -1 to start indexing at 0
    y_indices = jnp.digitize(coords[:, 1], spatial_bin_edges) - 1

    number_of_bins = len(spatial_bin_edges) - 1

    # Clip the indices to the valid range
    x_indices = jnp.clip(x_indices, 0, number_of_bins - 1)
    y_indices = jnp.clip(y_indices, 0, number_of_bins - 1)

    # Flatten the 2D indices to 1D indices
    pixel_positions = x_indices + (number_of_bins * y_indices)
    return pixel_positions


def filter_particles_outside_aperture(
    coords: Float[Array, " n_stars 3"],
    spatial_bin_edges: Float[Array, " n_bins"],
) -> Float[Array, " n_stars"]:
    """Mask the particles that are outside the aperture.

    Returns a boolean mask that is True for particles that are inside the aperture and False for particles.

    Parameters
    ----------
    coords : jnp.array (n, 3)
        The particle coordinates.

    spatial_bin_edges : jnp.array
    The bin edges for the spatial bins.

    Returns
    -------
    mask : jnp.array
    A boolean mask that is True for particles that are inside the aperture and False for particles that are outside the aperture.
    """
    min_value = spatial_bin_edges.min()
    max_value = spatial_bin_edges.max()

    mask = (coords[:, 0] >= min_value) & (coords[:, 0] <= max_value)
    mask &= (coords[:, 1] >= min_value) & (coords[:, 1] <= max_value)

    # Filter out all the particles that are outside the aperture

    return mask


# TODO: there is a better way to to this without loops
def restructure_data(masses, metallicities, ages, indices, num_pixels):
    # Calculate the number of particles per pixel
    # particle_count = jnp.bincount(indices, minlength=num_pixels)
    particle_count = jnp.bincount(indices, length=num_pixels)

    # Determine the maximum number of particles in any pixel
    max_particles = particle_count.max().astype(int)

    # Initialize structured arrays
    masses_structured = jnp.zeros((num_pixels, max_particles))
    metallicities_structured = jnp.zeros((num_pixels, max_particles))
    ages_structured = jnp.zeros((num_pixels, max_particles))

    # Process each pixel
    for i in range(num_pixels):
        # Find the indices of particles in this pixel
        particle_indices = jnp.flatnonzero(indices == i)
        num_particles_in_pixel = particle_indices.size

        # Update structured arrays with data from these particles
        if num_particles_in_pixel > 0:  # Only update if there are particles
            masses_structured = masses_structured.at[i, :num_particles_in_pixel].set(
                masses[particle_indices]
            )
            metallicities_structured = metallicities_structured.at[
                i, :num_particles_in_pixel
            ].set(metallicities[particle_indices])
            ages_structured = ages_structured.at[i, :num_particles_in_pixel].set(
                ages[particle_indices]
            )

    return masses_structured, metallicities_structured, ages_structured
