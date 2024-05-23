from typing import Callable

import jax

from rubix.logger import get_logger
from rubix.spectra.ssp.factory import get_ssp_template


def get_ssp(config: dict):
    # Check if field exists
    if "ssp" not in config:
        raise ValueError("Configuration does not contain 'ssp' field")

    # Check if template exists
    if "template" not in config["ssp"]:
        raise ValueError("Configuration does not contain 'template' field")
    # Check if name exists
    if "name" not in config["ssp"]["template"]:
        raise ValueError("Configuration does not contain 'name' field")
    ssp = get_ssp_template(config["ssp"]["template"]["name"])

    return ssp


def get_lookup(config: dict) -> Callable:
    """Loads the SSP template defined in the configuration and returns the lookup function for the template.

    The lookup function is a function that takes in the metallicity and age of a star and returns the spectrum of the star.
    This is later used to vmap over the stars metallicities and ages, and pmap over multiple GPUs.
    """
    logger_config = config.get("logger", None)
    logger = get_logger(logger_config)

    ssp = get_ssp(config)

    # Check if method is defined
    if "method" not in config["ssp"]:
        logger.debug("Method not defined, using default method: cubic")
        method = "cubic"
    else:
        logger.debug(f"Using method defined in config: {config['ssp']['method']}")
        method = config["ssp"]["method"]
        
    lookup = ssp.get_lookup(method=method)
    return lookup


def get_lookup_vmap(config: dict) -> Callable:
    """
    Get the lookup function for the SSP template defined in the configuration

    Loads the SSP template defined in the configuration and returns the lookup function for the template,
    vmapped over the stars metallicities and ages.
    """
    lookup = get_lookup(config)
    lookup_vmap = jax.vmap(lookup, in_axes=(0, 0))
    return lookup_vmap


def get_lookup_pmap(config: dict) -> Callable:
    """
    Get the pmap version of the lookup function for the SSP template defined in the configuration.
    """
    lookup_vmap = get_lookup_vmap(config)
    lookup_pmap = jax.pmap(lookup_vmap, in_axes=(0, 0))  # type: ignore
    return lookup_pmap
