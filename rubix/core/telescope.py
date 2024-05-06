from jaxtyping import Float, Array
from rubix.cosmology import RubixCosmology
from rubix.telescope.utils import calculate_spatial_bin_edges, square_spaxel_assignment
from rubix.telescope.base import BaseTelescope
from rubix.telescope.factory import TelescopeFactory
from jax.tree_util import Partial
from .cosmology import get_cosmology


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


def get_spaxel_assignment(config: dict) -> Float[Array, " n_stars"]:
    """Get the spaxel assignment function based on the configuration."""
    telescope = get_telescope(config)
    if telescope.pixel_type not in ["square"]:
        raise ValueError(f"Pixel type {telescope.pixel_type} not supported")
    spatial_bin_edges = get_spatial_bin_edges(config)
    return Partial(square_spaxel_assignment, spatial_bin_edges=spatial_bin_edges)  # type: ignore
