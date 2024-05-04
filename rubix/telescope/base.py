from jaxtyping import Float, Array
import equinox as eqx
from rubix.cosmology import RubixCosmology
from .utils import calculate_spatial_bin_edges, square_spaxel_assignment


class BaseTelescope(eqx.Module):
    fov: float
    spatial_res: float
    wave_range: Float[Array, "2"]  # upper and lower limits
    wave_res: float
    lsf_fwhm: float
    signal_to_noise: float
    wave_centre: float
    sbin: int
    aperture_region: Float[Array, " sbin*sbin"]
    pixel_type: str
    name: str = "BaseTelescope"

    def bin_particles(
        self,
        coords: Float[Array, " n_stars 3"],
        galaxy_dist_z: float,
        cosmology: RubixCosmology,
    ) -> Float[Array, " n_stars"]:
        """Bin the particle coordinates into a 2D image with the given bin edges based on the telescope configuration.

        This function takes the particle coordinates and bins them into a 2D image with the given bin edges.
        The binning is done by digitizing the x and y coordinates of the particles and then calculating the
        flat indices of the 2D image.

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
            fov=self.fov,
            spatial_bins=self.sbin,
            dist_z=galaxy_dist_z,
            cosmology=cosmology,
        )

        # Check which pixel type the telescope uses
        if self.pixel_type == "square":

            return square_spaxel_assignment(coords, spatial_bin_edges)

        else:
            raise ValueError(f"Pixel type {self.pixel_type} not supported")
