from typing import Callable

import jax
import jax.numpy as jnp

from rubix import config as rubix_config
from rubix.logger import get_logger
from rubix.spectra.ifu import (
    cosmological_doppler_shift,
    resample_spectrum,
    resample_spectrum_gas,
    velocity_doppler_shift,
    calculate_cube,
)

from .ssp import get_lookup_interpolation_pmap, get_ssp
from .telescope import get_telescope
from rubix.spectra.cloudy.grid import CloudyGasLookup
from rubix.spectra.cue.grid import CueGasLookup


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

        # Ensure metallicity and age are arrays and reshape them to be at least 1-dimensional
        age_data = jax.device_get(rubixdata.stars.age)
        metallicity_data = jax.device_get(rubixdata.stars.metallicity)
        # Ensure they are not scalars or empty; convert to 1D arrays if necessary
        age = jnp.atleast_1d(age_data)
        metallicity = jnp.atleast_1d(metallicity_data)

        spectra = lookup_interpolation_pmap(
            # rubixdata.stars.metallicity, rubixdata.stars.age
            metallicity,
            age,
        )  # * inputs["mass"]
        logger.debug(f"Calculation Finished! Spectra shape: {spectra.shape}")
        spectra_jax = jnp.array(spectra)
        rubixdata.stars.spectra = spectra_jax
        # setattr(rubixdata.gas, "spectra", spectra)
        # jax.debug.print("Calculate Spectra: Spectra {}", spectra)
        return rubixdata

    return calculate_spectra


def get_scale_spectrum_by_mass(config: dict) -> Callable:
    """Returns a function that scales the spectra by the mass of the stars or gas or both"""

    logger = get_logger(config.get("logger", None))

    def scale_spectrum_by_mass(rubixdata: object) -> object:
        logger.info("Scaling Spectra by Mass...")

        cube_type = config["data"]["args"].get("cube_type", [])

        if "stars" in cube_type:
            logger.info("Scaling Stars Spectra by Mass...")
            mass = jnp.expand_dims(rubixdata.stars.mass, axis=-1)
            spectra_mass = rubixdata.stars.spectra * mass
            setattr(rubixdata.stars, "spectra", spectra_mass)

        if "gas" in cube_type:
            logger.info("Scaling Gas Spectra by Mass...")
            mass = jnp.expand_dims(rubixdata.gas.mass, axis=-1)
            spectra_mass = rubixdata.gas.spectra * mass
            setattr(rubixdata.gas, "spectra", spectra_mass)

        return rubixdata

    return scale_spectrum_by_mass


# Vectorize the resample_spectrum function
def get_resample_spectrum_vmap(target_wavelength) -> Callable:
    logger = get_logger(rubix_config.get("logger", None))

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


# Vectorize the resample_spectrum function
def get_resample_spectrum_vmap_gas(target_wavelength) -> Callable:
    def resample_spectrum_vmap_gas(initial_spectrum, initial_wavelength):
        return resample_spectrum_gas(
            initial_spectrum=initial_spectrum,
            initial_wavelength=initial_wavelength,
            target_wavelength=target_wavelength,
        )

    return jax.vmap(resample_spectrum_vmap_gas, in_axes=(0, 0))


# Parallelize the vectorized function across devices
def get_resample_spectrum_pmap_gas(target_wavelength) -> Callable:
    vmapped_resample_spectrum_gas = get_resample_spectrum_vmap_gas(target_wavelength)
    return jax.pmap(vmapped_resample_spectrum_gas)


def get_velocities_doppler_shift_vmap(
    ssp_wave: jax.Array, velocity_direction: str
) -> Callable:
    def func(velocity):
        return velocity_doppler_shift(
            wavelength=ssp_wave, velocity=velocity, direction=velocity_direction
        )

    return jax.vmap(func, in_axes=0)


def get_velocities_doppler_shift_vmap_gas(
    cue_wave: jax.Array, velocity_direction: str
) -> Callable:
    def func(velocity):
        if velocity.ndim == 1 and velocity.shape[0] == 3:
            velocity = velocity.reshape(1, 3)
        return velocity_doppler_shift(
            wavelength=cue_wave, velocity=velocity, direction=velocity_direction
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

    cube_type = config["data"]["args"].get("cube_type", [])

    if "stars" in cube_type:
        # Get the SSP grid to doppler shift the wavelengths
        ssp = get_ssp(config)

        # Doppler shift the SSP wavelength based on the cosmological distance of the observed galaxy
        ssp_wave = cosmological_doppler_shift(
            z=galaxy_redshift, wavelength=ssp.wavelength
        )
        logger.debug(f"SSP Wave: {ssp_wave.shape}")

        # Function to Doppler shift the wavelength based on the velocity of the stars particles
        doppler_shift = get_velocities_doppler_shift_vmap(ssp_wave, velocity_direction)

    if "gas" in cube_type:
        """
        filepath = "../rubix/spectra/cloudy/templates/UVB_plus_CMB_strongUV_line_emissivities.dat"
        cloudy = CloudyGasLookup(filepath)

        cloudy_wave = cloudy.get_wavelengthrange()
        logger.debug(f"CLOUDY Wave: {cloudy_wave.shape}")

        # Function to Doppler shift the wavelength based on the velocity of the gas particles
        doppler_shift_cloudy = get_velocities_doppler_shift_vmap_gas(
            cloudy_wave, velocity_direction
        )
        """
        # Get the CUE wavelength grid to doppler shift the wavelengths
        cue = CueGasLookup(config)
        cue_wavelength = cue.get_wavelengthrange()
        # logger.debug(f"CUE Wave: {cue_wavelength}")

        # Doppler shift the CUE wavelength based on the cosmological distance of the observed galaxy
        cue_wave = cosmological_doppler_shift(
            z=galaxy_redshift, wavelength=cue_wavelength
        )
        logger.debug(f"CUE Wave: {cue_wave.shape}")
        # logger.debug(f"cue_wave: {cue_wave}")
        # logger.debug(f"velocity_direction: {velocity_direction}")
        # Function to Doppler shift the wavelength based on the velocity of the gas particles
        doppler_shift_cue = get_velocities_doppler_shift_vmap_gas(
            cue_wave, velocity_direction
        )

    def doppler_shift_and_resampling(rubixdata: object) -> object:
        cube_type = config["data"]["args"].get("cube_type", [])
        logger.debug(f"Cube Type: {cube_type}")

        if "stars" in cube_type:
            # Doppler shift the SSP Wavelengths based on the velocity of the stars
            doppler_shifted_ssp_wave = doppler_shift(rubixdata.stars.velocity)
            logger.info("Doppler shifting and resampling stellar spectra...")
            logger.debug(f"Doppler Shifted SSP Wave: {doppler_shifted_ssp_wave.shape}")
            logger.debug(f"Telescope Wave Seq: {telescope.wave_seq.shape}")
            # Function to resample the spectrum to the telescope wavelength grid
            resample_spectrum_pmap = get_resample_spectrum_pmap(telescope_wavelenght)
            spectrum_resampled = resample_spectrum_pmap(
                rubixdata.stars.spectra, doppler_shifted_ssp_wave
            )
            setattr(rubixdata.stars, "spectra", spectrum_resampled)

        if "gas" in cube_type:
            # Doppler shift the Cloudy Wavelengths based on the velocity of the gas particles
            # logger.debug(f"Velocity: {rubixdata.gas.velocity}")
            logger.debug(f"Gas spectra shape: {rubixdata.gas.spectra.shape}")
            # Reshape the gas spectra to (1, 100, 1000)
            rubixdata.gas.spectra = jnp.expand_dims(rubixdata.gas.spectra, axis=0)
            logger.debug(
                f"Processed rubixdata.gas.spectra shape: {rubixdata.gas.spectra.shape}"
            )
            rubixdata.gas.velocity = jnp.expand_dims(rubixdata.gas.velocity, axis=0)
            doppler_shifted_cue_wave = doppler_shift_cue(rubixdata.gas.velocity)
            # logger.debug(f"Doppler Shifted CUE Wave: {doppler_shifted_cue_wave}")
            if doppler_shifted_cue_wave.shape[1] == 1:
                # Reshape to remove the single-dimensional entry
                doppler_shifted_cue_wave = doppler_shifted_cue_wave.reshape(
                    doppler_shifted_cue_wave.shape[0], -1
                )

            logger.info("Doppler shifting and resampling gas spectra...")
            logger.debug(f"Doppler shifted CUE Wave: {doppler_shifted_cue_wave.shape}")
            logger.debug(f"Telescope Wave Seq: {telescope.wave_seq.shape}")
            logger.debug(f"Spectra before resampling: {rubixdata.gas.spectra}")
            # Function to resample the spectrum to the telescope wavelength grid
            resample_spectrum_pmap = get_resample_spectrum_pmap_gas(
                telescope_wavelenght
            )
            spectrum_resampled_gas = resample_spectrum_pmap(
                rubixdata.gas.spectra, doppler_shifted_cue_wave
            )
            spectrum_resampled_gas = jnp.nan_to_num(
                spectrum_resampled_gas, nan=0.0, posinf=0.0, neginf=0.0
            )
            setattr(rubixdata.gas, "spectra", spectrum_resampled_gas)
            logger.debug(f"Spectra after resampling: {rubixdata.gas.spectra}")

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
        cube_type = config["data"]["args"].get("cube_type", [])
        logger.debug(f"Cube Type: {cube_type}")

        if "stars" in cube_type:
            logger.info("Calculating Data Cube Stars...")
            logger.debug(f"Star spectra shape: {rubixdata.stars.spectra.shape}")
            logger.debug(
                f"Star pixel assignment shape: {rubixdata.stars.pixel_assignment.shape}"
            )
            ifu_cubes = calculate_cube_pmap(
                spectra=rubixdata.stars.spectra,
                spaxel_index=rubixdata.stars.pixel_assignment,
            )
            datacube = jnp.sum(ifu_cubes, axis=0)
            logger.debug(f"Datacube Shape: {datacube.shape}")
            # logger.debug(f"This is the datacube: {datacube}")
            datacube_jax = jnp.array(datacube)
            setattr(rubixdata.stars, "datacube", datacube_jax)
            # rubixdata.stars.datacube = datacube
            return rubixdata

        if "gas" in cube_type:
            logger.info("Calculating Data Cube Gas...")
            logger.debug(f"Gas spectra shape: {rubixdata.gas.spectra.shape}")
            # if rubixdata.gas.spectra.shape[0] == 1:
            #    rubixdata.gas.spectra = jnp.squeeze(rubixdata.gas.spectra, axis=0)
            if rubixdata.gas.pixel_assignment.shape[0] != 1:
                rubixdata.gas.pixel_assignment = jnp.expand_dims(
                    rubixdata.gas.pixel_assignment, axis=0
                )
            logger.debug(
                f"Processed rubixdata.gas.spectra shape: {rubixdata.gas.spectra.shape}"
            )
            logger.debug(f"Gas spectra shape: {rubixdata.gas.spectra.shape}")
            logger.debug(
                f"Gas pixel assignment shape: {rubixdata.gas.pixel_assignment.shape}"
            )
            ifu_cubes_gas = calculate_cube_pmap(
                spectra=rubixdata.gas.spectra,
                spaxel_index=rubixdata.gas.pixel_assignment,
            )
            datacube_gas = jnp.sum(ifu_cubes_gas, axis=0)
            logger.debug(f"Datacube Shape: {datacube_gas.shape}")
            # logger.debug(f"This is the datacube: {ifu_cube_gas}")
            datacube_gas_jax = jnp.array(datacube_gas)
            setattr(rubixdata.gas, "datacube", datacube_gas_jax)
            # rubixdata.gas.datacube = datacube
            return rubixdata

    return calculate_datacube
