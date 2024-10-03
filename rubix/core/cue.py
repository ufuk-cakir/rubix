import jax.numpy as jnp
import jax
from rubix.spectra.cue.grid import CueGasLookup
from rubix.logger import get_logger


def get_gas_emission(config: dict):

    if "gas" not in config["data"]["args"]["particle_type"]:
        raise ValueError("No gas particle data in this galaxy")

    if "gas" not in config["data"]["args"]["cube_type"]:
        raise ValueError("Not specified in the config to calculate gas emission cube")

    logger = get_logger()

    def gas_emission(rubixdata: object) -> object:
        """Calculate gas emission lines and gas continuum emission."""
        logger.info("Calculation gas emission...")

        CueClass = CueGasLookup(config)

        rubixdata = CueClass.get_gas_emission_flux(rubixdata)
        rubixdata = jax.block_until_ready(rubixdata)

        logger.debug("Completed gas emission calculation: %s", rubixdata)
        return rubixdata

    return gas_emission
