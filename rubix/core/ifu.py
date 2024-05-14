from .ssp import get_lookup_pmap
from rubix.logger import get_logger
import jax


def get_calculate_spectra(config: dict):

    logger = get_logger(config.get("logger", None))
    lookup_pmap = get_lookup_pmap(config)

    def calculate_spectra(inputs: dict[str, jax.Array]):
        logger.debug("Calculating IFU cube...")
        spectra = lookup_pmap(inputs["metallicity"], inputs["age"]) * inputs["mass"]

        inputs["spectra"] = spectra
        return inputs

    return calculate_spectra
