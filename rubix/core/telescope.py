import jax.numpy as jnp
from jaxtyping import Float, Array
from rubix.telescope.utils import (
    calculate_spatial_bin_edges,
    square_spaxel_assignment,
    mask_particles_outside_aperture,
)
from rubix.telescope.base import BaseTelescope
from rubix.telescope.factory import TelescopeFactory
from .cosmology import get_cosmology
from typing import Callable


def get_telescope(config: dict) -> BaseTelescope:
    """Get the telescope object based on the configuration."""
    # TODO: this currently only loads telescope that are supported.
    # add support for custom telescopes
    factory = TelescopeFactory()
    telescope = factory.create_telescope(config["telescope"]["name"])
    return telescope


def get_spatial_bin_edges(config: dict) -> Float[Array, " n_bins"]:
    """Get the spatial bin edges based on the configuration."""
    telescope = get_telescope(config)
    galaxy_dist_z = config["galaxy"]["dist_z"]
    cosmology = get_cosmology(config)
    # Calculate the spatial bin edges
    # TODO: check if we need the spatial bin size somewhere? For now we dont use it
    spatial_bin_edges, spatial_bin_size = calculate_spatial_bin_edges(
        fov=telescope.fov,
        spatial_bins=telescope.sbin,
        dist_z=galaxy_dist_z,
        cosmology=cosmology,
    )

    return spatial_bin_edges


def get_spaxel_assignment(config: dict) -> Callable:
    """Get the spaxel assignment function based on the configuration."""
    telescope = get_telescope(config)
    if telescope.pixel_type not in ["square"]:
        raise ValueError(f"Pixel type {telescope.pixel_type} not supported")
    spatial_bin_edges = get_spatial_bin_edges(config)

    def spaxel_assignment(rubixdata: object) -> object:
        if "stars" in config["data"]["args"]["particle_type"]:
            pixel_assignment = square_spaxel_assignment(
                rubixdata.stars.coords, spatial_bin_edges
            )
            rubixdata.stars.pixel_assignment = pixel_assignment
            rubixdata.stars.spatial_bin_edges = spatial_bin_edges
        if "gas" in config["data"]["args"]["particle_type"]:
            pixel_assignment = square_spaxel_assignment(
                rubixdata.gas.coords, spatial_bin_edges
            )
            rubixdata.gas.pixel_assignment = pixel_assignment
            rubixdata.gas.spatial_bin_edges = spatial_bin_edges

        return rubixdata

    return spaxel_assignment


def get_filter_particles(config: dict):
    """Get the function to filter particles outside the aperture."""
    spatial_bin_edges = get_spatial_bin_edges(config)

    def filter_particles(rubixdata: object) -> object:
        if "stars" in config["data"]["args"]["particle_type"]:
            mask = mask_particles_outside_aperture(rubixdata.stars.coords, spatial_bin_edges)

            # input_data["coords"] = input_data["coords"][mask]
            # input_data["velocities"] = input_data["velocities"][mask]
            #input_data["mass"] = jnp.where(mask, input_data["mass"], 0)
            #input_data["age"] = jnp.where(mask, input_data["age"], 0)
            #input_data["metallicity"] = jnp.where(mask, input_data["metallicity"], 0)
            attributes = [attr for attr in dir(rubixdata.stars) if not attr.startswith('__') and attr not in ('coords', 'velocity')]
            for attr in attributes:
                rubixdata.stars.__setattr__(attr, jnp.where(mask, rubixdata.stars.__getattribute__(attr), 0))
            rubixdata.stars.mask = mask

        if "gas" in config["data"]["args"]["particle_type"]:
            mask = mask_particles_outside_aperture(rubixdata.gas.coords, spatial_bin_edges)

            attributes = [attr for attr in dir(rubixdata.gas) if not attr.startswith('__') and attr not in ('coords', 'velocity')]
            for attr in attributes:
                rubixdata.gas.__setattr__(attr, jnp.where(mask, rubixdata.gas.__getattribute__(attr), 0))
            rubixdata.gas.mask = mask

        return rubixdata

    return filter_particles


# def get_split_data(config: dict, n_particles) -> Callable:
#     telescope = get_telescope(config)
#     n_pixels = telescope.sbin**2
#
#     def split_data(input_data: dict) -> dict:
#         # Split the data into two parts
#
#         masses, metallicity, ages = restructure_data(
#             input_data["mass"],
#             input_data["metallicity"],
#             input_data["age"],
#             input_data["pixel_assignment"],
#             n_pixels,
#             # n_particles,
#         )
#
#         # Reshape the data to match the number of GPUs
#
#         input_data["masses"] = masses
#         input_data["metallicity"] = metallicity
#         input_data["age"] = ages
#
#         return input_data
#
#     return split_data
