from abc import ABC, abstractmethod
import jax.numpy as jnp
from jax import jit

# TODO do I need this as a hyperparemter? 
# such that the user can choose the cosmology
from rubix.cosmology import PLANCK15 as rubixcosmo
from .utils import assign_particles_to_pixel_positions

class BaseTelescope(ABC):
    def __init__(
        self,
        name = None,
        *,
        fov,
        spatial_res,
        wave_range,
        wave_res,
        lsf_fwhm,
        signal_to_noise,
        wave_centre,
        sbin,
        aperture_region
    ):
        self.name = name
        self.fov = fov
        self.spatial_res = spatial_res
        self.wave_range = wave_range
        self.wave_res = wave_res
        self.lsf_fwhm = lsf_fwhm
        self.signal_to_noise = signal_to_noise or jnp.inf  # Default to infinity if None
        self.wave_centre = wave_centre
        self.sbin = sbin or int(jnp.floor(self.fov / self.spatial_res))
        self.aperture_region = aperture_region


    # TODO here we need the redshift distance where we observe the galaxy
    # Need to think about where to pass this information
    def spaxel_assignment(self, galaxy_distance_z):
        
        D_A = rubixcosmo.angular_diameter_distance_to_z(galaxy_distance_z) # in Mpc
        # TODO check this
        ang_size = D_A* (jnp.pi/(180*3600)) *1e3 # in kpc/arcsec
        aperture_size = self.fov * ang_size
        
        spatial_bin_size = aperture_size / self.sbin
        spatial_bin_edges = jnp.arange(
        -(self.sbin * spatial_bin_size) / 2,
        (self.sbin * spatial_bin_size) / 2,
        spatial_bin_size,
        )  # spatial bin break positions
        
        
        number_of_bins = self.sbin
        
        # TODO is it okay to use lambda like this?
        func = lambda coords: assign_particles_to_pixel_positions(coords, spatial_bin_edges, number_of_bins)
        return jit(func)
    

    def __str__(self) -> str:
        dict = self.__dict__
        return_str = f"{self.name}: \n"
        dict.pop("name")
        for key, value in dict.items():
            return_str += f" {key}={value}, \n"
        return_str = return_str[:-2] 
        return return_str

            