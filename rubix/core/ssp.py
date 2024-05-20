from rubix.spectra.ssp.factory import get_ssp_template
from typing import Callable
from rubix.logger import get_logger
import jax


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

    Loads the SSP template defined in the configuration and returns the lookup function for the template.
    """
    lookup = get_lookup(config)
    lookup_vmap = jax.vmap(lookup, in_axes=(0, 0))
    return lookup_vmap


def get_lookup_pmap(config: dict) -> Callable:
    lookup_vmap = get_lookup_vmap(config)
    lookup_pmap = jax.pmap(lookup_vmap, in_axes=(0, 0))
    return lookup_pmap
