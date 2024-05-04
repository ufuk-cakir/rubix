from jaxtyping import Float, Array
from rubix.cosmology import RubixCosmology
from rubix.telescope.utils import calculate_spatial_bin_edges, square_spaxel_assignment
from rubix.telescope.base import BaseTelescope
from jax.tree_util import Partial


def get_spaxel_assignment(
    telescope: BaseTelescope,
    coords: Float[Array, " n_stars 3"],
    galaxy_dist_z: float,
    cosmology: RubixCosmology,
) -> Float[Array, " n_stars"]:
    """Get the spaxel assignment function based on the telescope configuration.


    Parameters
    ----------
    coords : jnp.array (n, 3)
        The particle coordinates.

    cosmology : RubixCosmology
        The cosmology object.

    dist_z : float
        The redshift of the particles.
    """

    # Calculate the spatial bin edges
    # TODO check if we need the spatial bin size somewhere? For now we dont use it
    spatial_bin_edges, spatial_bin_size = calculate_spatial_bin_edges(
        fov=telescope.fov,
        spatial_bins=telescope.sbin,
        dist_z=galaxy_dist_z,
        cosmology=cosmology,
    )

    # Check which pixel type the telescope uses
    if telescope.pixel_type == "square":
        # Return a function, so we can use it in the pipeline
        # partial function with spatial_bin_edges as a fixed argument
        # this way we can pass the coords only

        # we use Partial for this
        return Partial(square_spaxel_assignment, spatial_bin_edges=spatial_bin_edges)  # type: ignore
    else:
        raise ValueError(f"Pixel type {telescope.pixel_type} not supported")
