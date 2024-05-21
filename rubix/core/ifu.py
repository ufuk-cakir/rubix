from .ssp import get_lookup_pmap
from rubix.logger import get_logger
import jax
import jax.numpy as jnp
from rubix import config as rubix_config
from rubix.spectra.ifu import (
    velocity_doppler_shift,
    cosmological_doppler_shift,
    resample_spectrum,
)
from .telescope import get_telescope
from .ssp import get_ssp
from .data import reshape_array


def get_calculate_spectra(config: dict):

    logger = get_logger(config.get("logger", None))
    lookup_pmap = get_lookup_pmap(config)

    def calculate_spectra(inputs: dict[str, jax.Array]):
        logger.info("Calculating IFU cube...")
        logger.debug(
            f"Input shapes: Metallicity: {inputs['metallicity'].shape}, Age: {inputs['age'].shape}"
        )
        spectra = lookup_pmap(inputs["metallicity"], inputs["age"])  # * inputs["mass"]
        logger.debug(f"Calculation Finished! Spectra shape: {spectra.shape}")
        inputs["spectra"] = spectra
        # jax.debug.print("Calculate Spectra: Spectra {}", spectra)
        return inputs

    return calculate_spectra


def get_scale_spectrum_by_mass(config: dict):

    logger = get_logger(config.get("logger", None))

    def scale_spectrum_by_mass(inputs: dict[str, jax.Array]):
        logger.info("Scaling Spectra by Mass...")
        mass = jnp.expand_dims(inputs["mass"], axis=-1)
        inputs["spectra"] = inputs["spectra"] * mass
        # jax.debug.print("mass mult: Spectra {}", inputs["spectra"])
        return inputs

    return scale_spectrum_by_mass


# def get_resample_spectrum_vmap(initial_wavelength, target_wavelength):
#     def resample_spectrum_vmap(spectra: jax.Array) -> jax.Array:
#         return resample_spectrum(
#             initial_spectrum=spectra,
#             initial_wavelength=initial_wavelength,
#             target_wavelength=target_wavelength,
#         )
#
#     return jax.vmap(resample_spectrum_vmap)
#
#
# def get_resample_spectrum_pmap(initial_wavelength, target_wavelength):
#     vmapped_resample_spectrum = get_resample_spectrum_vmap(
#         initial_wavelength, target_wavelength
#     )
#     return jax.pmap(vmapped_resample_spectrum)


# Vectorize the resample_spectrum function
def get_resample_spectrum_vmap(target_wavelength):
    def resample_spectrum_vmap(initial_spectrum, initial_wavelength):
        return resample_spectrum(
            initial_spectrum=initial_spectrum,
            initial_wavelength=initial_wavelength,
            target_wavelength=target_wavelength,
        )

    return jax.vmap(resample_spectrum_vmap, in_axes=(0, 0))


# Parallelize the vectorized function across devices
def get_resample_spectrum_pmap(target_wavelength):
    vmapped_resample_spectrum = get_resample_spectrum_vmap(target_wavelength)
    return jax.pmap(vmapped_resample_spectrum)


def get_velocities_doppler_shift_vmap(ssp_wave: jax.Array, velocity_direction: str):
    def func(velocity):
        return velocity_doppler_shift(
            wavelength=ssp_wave, velocity=velocity, direction=velocity_direction
        )

    return jax.vmap(func, in_axes=0)


def get_doppler_shift_and_resampling(config: dict):

    logger = get_logger(config.get("logger", None))
    velocity_direction = rubix_config["ifu"]["doppler"]["velocity_direction"]
    galaxy_redshift = config["galaxy"]["dist_z"]
    telescope = get_telescope(config)
    telescope_wavelenght = telescope.wave_seq
    ssp = get_ssp(config)

    # Doppler shift the SSP wavelenght based on the cosmological distance of the observed galaxy
    ssp_wave = cosmological_doppler_shift(z=galaxy_redshift, wavelength=ssp.wavelength)
    logger.debug(f"SSP Wave: {ssp_wave.shape}")
    # Function to Doppler shift the wavelength based on the velocity of the stars particles
    doppler_shift = get_velocities_doppler_shift_vmap(ssp_wave, velocity_direction)

    def doppler_shift_and_resampling(
        inputs: dict[str, jax.Array]
    ) -> dict[str, jax.Array]:
        doppler_shifted_ssp_wave = doppler_shift(inputs["velocities"])
        logger.debug(f"Doppler Shifted SSP Wave: {doppler_shifted_ssp_wave.shape}")
        logger.debug(f"Telescope Wave Seq: {telescope.wave_seq.shape}")
        # Function to resample the spectrum to the telescope wavelength grid
        resample_spectrum_pmap = get_resample_spectrum_pmap(telescope_wavelenght)
        # jax.debug.print("doppler shifted ssp wave {}", doppler_shifted_ssp_wave)
        # jax.debug.print("Spectra before resampling {}", inputs["spectra"])
        spectrum_resampled = resample_spectrum_pmap(
            inputs["spectra"], doppler_shifted_ssp_wave
        )
        inputs["spectra"] = spectrum_resampled
        # jax.debug.print("doppler shift and resampl: Spectra {}", inputs["spectra"])
        return inputs

    return doppler_shift_and_resampling
