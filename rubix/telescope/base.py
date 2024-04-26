from jaxtyping import Float, Array
from typing import List
import equinox as eqx


class BaseTelescope(eqx.Module):
    fov: float
    spatial_res: float
    wave_range: Float[Array, "2"] # upper and lower limits
    wave_res: float
    lsf_fwhm: float
    signal_to_noise: float
    wave_centre: float
    sbin: int
    aperture_region: Float[Array, " sbin*sbin"]
    name: str = "BaseTelescope"

