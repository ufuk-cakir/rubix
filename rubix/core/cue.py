import jax.numpy as jnp
import jax
from rubix.spectra.cue.grid import CueGasLookup
from rubix.logger import get_logger
from rubix.core.telescope import get_telescope
from rubix.cosmology.base import BaseCosmology
from rubix.spectra.ifu import convert_luminoisty_to_flux_gas
from rubix import config as rubix_config


def preprocess_config(config: dict):
    """
    Perform any necessary pre-processing based on the config and return
    pre-processed data that can be passed to the JAX-compiled function.
    """
    # Initialize the logger
    logger = get_logger(config.get("logger", None))

    # Retrieve the telescope configuration
    telescope = get_telescope(config)
    spatial_res = telescope.spatial_res

    # Cosmology calculations
    logger.warning("Assuming Planck Cosmology for converting from luminosity to flux.")
    cosmo = BaseCosmology(0.3089, -1.0, 0.0, 0.6774)
    observation_lum_dist = cosmo.luminosity_distance_to_z(config["galaxy"]["dist_z"])

    # Calculate the factor to convert luminosity to flux
    factor = convert_luminoisty_to_flux_gas(
        observation_lum_dist,
        observation_z=config["galaxy"]["dist_z"],
        pixel_size=spatial_res,
        CONSTANTS=rubix_config["constants"],
    )
    factor = jnp.float32(factor)

    return {
        "config": config,
        "telescope": telescope,
        "observation_lum_dist": observation_lum_dist,
        "factor": factor,
    }


def preprocess_cue(config: dict):
    """
    Perform any necessary pre-processing based on the config and return
    pre-processed data that can be passed to the JAX-compiled function.
    """
    pre_config = preprocess_config(config)
    CueClass = CueGasLookup(pre_config)

    # Perform any necessary lookup and pre-processing steps here
    preprocessed_lookup_data = CueClass  # For now, just passing the class itself
    return preprocessed_lookup_data


def get_gas_emission(config: dict):

    if "gas" not in config["data"]["args"]["particle_type"]:
        raise ValueError("No gas particle data in this galaxy")

    if "gas" not in config["data"]["args"]["cube_type"]:
        raise ValueError("Not specified in the config to calculate gas emission cube")

    logger = get_logger()

    # Perform the pre-processing step outside of the returned function
    preprocessed_lookup_data = preprocess_cue(config)

    def gas_emission(rubixdata: object) -> object:
        """Calculate gas emission lines and gas continuum emission."""
        logger.info("Calculation gas emission...")

        # Use the pre-processed lookup data here
        rubixdata = preprocessed_lookup_data.get_gas_emission_flux(rubixdata)
        rubixdata = jax.block_until_ready(rubixdata)
        # rubixdata = jax.device_put(rubixdata)

        logger.debug("Completed gas emission calculation: %s", rubixdata)
        logger.debug(
            "test core module: temperature: %s", jnp.array(rubixdata.gas.temperature)
        )
        logger.debug(
            "test core module: continuum: %s", jnp.array(rubixdata.gas.continuum)
        )
        logger.debug(
            "test core module: emission: %s", jnp.array(rubixdata.gas.emission_spectra)
        )
        return rubixdata

    return gas_emission
