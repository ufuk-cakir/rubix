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
