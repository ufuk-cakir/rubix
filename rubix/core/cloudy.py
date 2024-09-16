from typing import Callable
import jax.numpy as jnp
from rubix.logger import get_logger


def get_gas_spectra(config: dict) -> Callable:
    logger = get_logger(config.get("logger", None))

    def gas_spectra(rubixdata: object) -> object:
        logger.info("Calculating Gas Spectra...")
        return rubixdata

    return gas_spectra


def get_remove_gas_spectra(config: dict) -> Callable:
    logger = get_logger(config.get("logger", None))

    def remove_gas_spectra(rubixdata: object) -> object:
        logger.info("Removing Gas Spectra...")
        nan_mask = jnp.isnan(rubixdata.gas.spectra)
        # Replace NaN values with 0
        cleaned_array = jnp.where(nan_mask, 0, rubixdata.gas.spectra)

        setattr(rubixdata.gas, "spectra", cleaned_array)

        return rubixdata

    return remove_gas_spectra
