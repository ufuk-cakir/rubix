from typing import Callable

import jax
import jax.numpy as jnp

from rubix import config as rubix_config
from rubix.logger import get_logger
from rubix.spectra.ifu import (
    cosmological_doppler_shift,
    resample_spectrum,
    velocity_doppler_shift,
    calculate_cube,
)

from .ssp import get_lookup_interpolation_pmap, get_ssp
from .telescope import get_telescope


def get_calculate_spectra(config: dict) -> Callable:
    """Returns a function that calculates the spectra of the stars

    The function get the lookup function that performs the lookup to the SSP model,
    and parallelizes the funciton across all GPUs
    """
    logger = get_logger(config.get("logger", None))
    lookup_interpolation_pmap = get_lookup_interpolation_pmap(config)

    def calculate_spectra(rubixdata: object) -> object:
        logger.info("Calculating IFU cube...")
        logger.debug(
            f"Input shapes: Metallicity: {len(rubixdata.stars.metallicity)}, Age: {len(rubixdata.stars.age)}"
        )
        #print(rubixdata.stars.metallicity)
        #print(rubixdata.stars.metallicity.shape)
        
            
        # Ensure metallicity and age are arrays and reshape them to be at least 1-dimensional
        age_data = jax.device_get(rubixdata.stars.age)
        metallicity_data = jax.device_get(rubixdata.stars.metallicity)
        
        # Ensure they are not scalars or empty; convert to 1D arrays if necessary
        age = jnp.atleast_1d(age_data)
        metallicity = jnp.atleast_1d(metallicity_data)

        # Debugging: Print shapes to verify
        #print(f"Metallicity shape: {metallicity.shape}, Age shape: {age.shape}")

        # Check if either is still a scalar or empty, which indicates a problem
        #if metallicity.shape == () or age.shape == ():
        #    raise ValueError("Metallicity or age is a scalar or empty, which is not expected.")
        #print(rubixdata.stars.metallicity.shape)
        #print(f"Metallicity shape before call: {metallicity.shape}, Age shape before call: {age.shape}")
        #metallicity = rubixdata.stars.metallicity
        #age = rubixdata.stars.age

        spectra = lookup_interpolation_pmap(
            #rubixdata.stars.metallicity, rubixdata.stars.age
            metallicity, age
        )  # * inputs["mass"]
        logger.debug(f"Calculation Finished! Spectra shape: {spectra.shape}")
        spectra_jax = jnp.array(spectra)
        rubixdata.stars.spectra = spectra_jax
        #setattr(rubixdata.gas, "spectra", spectra)
        # jax.debug.print("Calculate Spectra: Spectra {}", spectra)
        return rubixdata

    return calculate_spectra


def get_scale_spectrum_by_mass(config: dict) -> Callable:
    """Returns a function that scales the spectra by the mass of the stars"""

    logger = get_logger(config.get("logger", None))

    def scale_spectrum_by_mass(rubixdata: object) -> object:
        logger.info("Scaling Spectra by Mass...")
        mass = jnp.expand_dims(rubixdata.stars.mass, axis=-1)
        #rubixdata.stars.spectra = rubixdata.stars.spectra * mass
        spectra_mass = rubixdata.stars.spectra * mass
        setattr(rubixdata.stars, "spectra", spectra_mass)
        # jax.debug.print("mass mult: Spectra {}", inputs["spectra"])
        return rubixdata

    return scale_spectrum_by_mass


# Vectorize the resample_spectrum function
def get_resample_spectrum_vmap(target_wavelength) -> Callable:
    def resample_spectrum_vmap(initial_spectrum, initial_wavelength):
        return resample_spectrum(
            initial_spectrum=initial_spectrum,
            initial_wavelength=initial_wavelength,
            target_wavelength=target_wavelength,
        )

    return jax.vmap(resample_spectrum_vmap, in_axes=(0, 0))


# Parallelize the vectorized function across devices
def get_resample_spectrum_pmap(target_wavelength) -> Callable:
    vmapped_resample_spectrum = get_resample_spectrum_vmap(target_wavelength)
    return jax.pmap(vmapped_resample_spectrum)


def get_velocities_doppler_shift_vmap(
    ssp_wave: jax.Array, velocity_direction: str
) -> Callable:
    def func(velocity):
        return velocity_doppler_shift(
            wavelength=ssp_wave, velocity=velocity, direction=velocity_direction
        )

    return jax.vmap(func, in_axes=0)


def get_doppler_shift_and_resampling(config: dict) -> Callable:
    logger = get_logger(config.get("logger", None))

    # The velocity component of the stars that is used to doppler shift the wavelength
    velocity_direction = rubix_config["ifu"]["doppler"]["velocity_direction"]

    # The redshift at which the user wants to observe the galaxy
    galaxy_redshift = config["galaxy"]["dist_z"]

    # Get the telescope wavelength bins
    telescope = get_telescope(config)
    telescope_wavelenght = telescope.wave_seq

    # Get the SSP grid to doppler shift the wavelengths
    ssp = get_ssp(config)

    # Doppler shift the SSP wavelenght based on the cosmological distance of the observed galaxy
    ssp_wave = cosmological_doppler_shift(z=galaxy_redshift, wavelength=ssp.wavelength)
    logger.debug(f"SSP Wave: {ssp_wave.shape}")

    # Function to Doppler shift the wavelength based on the velocity of the stars particles
    # This binds the velocity direction, such that later we only need the velocity during the pipeline
    doppler_shift = get_velocities_doppler_shift_vmap(ssp_wave, velocity_direction)

    def doppler_shift_and_resampling(rubixdata: object) -> object:
        if rubixdata.stars.velocity is not None:
            # Doppler shift the SSP Wavelengths based on the velocity of the stars
            doppler_shifted_ssp_wave = doppler_shift(rubixdata.stars.velocity)
            logger.info("Doppler shifting and resampling stellar spectra...")
            logger.debug(f"Doppler Shifted SSP Wave: {doppler_shifted_ssp_wave.shape}")
            logger.debug(f"Telescope Wave Seq: {telescope.wave_seq.shape}")
            # Function to resample the spectrum to the telescope wavelength grid
            resample_spectrum_pmap = get_resample_spectrum_pmap(telescope_wavelenght)
            # jax.debug.print("doppler shifted ssp wave {}", doppler_shifted_ssp_wave)
            # jax.debug.print("Spectra before resampling {}", inputs["spectra"])
            spectrum_resampled = resample_spectrum_pmap(
                rubixdata.stars.spectra, doppler_shifted_ssp_wave
            )
            #rubixdata.stars.spectra = spectrum_resampled
            setattr(rubixdata.stars, "spectra", spectrum_resampled)
            # jax.debug.print("doppler shift and resampl: Spectra {}", inputs["spectra"])

        if rubixdata.gas.velocity is not None:
            # Doppler shift the SSP Wavelengths based on the velocity of the gas particles
            doppler_shifted_ssp_wave = doppler_shift(rubixdata.gas.velocity)
            logger.info("Doppler shifting and resampling gas spectra...")
            logger.debug(f"Doppler Shifted SSP Wave: {doppler_shifted_ssp_wave.shape}")
            logger.debug(f"Telescope Wave Seq: {telescope.wave_seq.shape}")
            # Function to resample the spectrum to the telescope wavelength grid
            resample_spectrum_pmap = get_resample_spectrum_pmap(telescope_wavelenght)
            spectrum_resampled = resample_spectrum_pmap(
                rubixdata.gas.spectra, doppler_shifted_ssp_wave
            )

        return rubixdata

    return doppler_shift_and_resampling


def get_calculate_datacube(config: dict) -> Callable:
    logger = get_logger(config.get("logger", None))
    telescope = get_telescope(config)
    num_spaxels = telescope.sbin

    # Bind the num_spaxels to the function
    calculate_cube_fn = jax.tree_util.Partial(calculate_cube, num_spaxels=num_spaxels)
    calculate_cube_pmap = jax.pmap(calculate_cube_fn)

    def calculate_datacube(rubixdata: object) -> object:
        logger.info("Calculating Data Cube...")
        ifu_cubes = calculate_cube_pmap(
            spectra=rubixdata.stars.spectra, spaxel_index=rubixdata.stars.pixel_assignment
        )
        datacube = jnp.sum(ifu_cubes, axis=0)
        logger.debug(f"Datacube Shape: {datacube.shape}")
        logger.debug(f"This is the datacube: {datacube}")
        datacube_jax = jnp.array(datacube)
        setattr(rubixdata.stars, "datacube", datacube_jax)
        #rubixdata.stars.datacube = datacube
        return rubixdata

    return calculate_datacube
