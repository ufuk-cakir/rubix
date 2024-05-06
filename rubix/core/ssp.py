from rubix.spectra.ssp.factory import get_ssp_template
from typing import Callable
from rubix.logger import get_logger


def get_lookup(config: dict) -> Callable:
    """
    Get the lookup function for the SSP template defined in the configuration

    Loads the SSP template defined in the configuration and returns the lookup function for the template.
    """
    logger_config = config.get("logger", None)
    logger = get_logger(logger_config)
    # Check if field exists
    if "ssp" not in config:
        raise ValueError("Configuration does not contain 'ssp' field")

    # Check if template exists
    if "template" not in config["ssp"]:
        raise ValueError("Configuration does not contain 'template' field")
    # Check if name exists
    if "name" not in config["ssp"]["template"]:
        raise ValueError("Configuration does not contain 'name' field")

    # Get the ssp template
    logger.debug(f"Getting SSP template: {config['ssp']['template']['name']}")
    ssp = get_ssp_template(config["ssp"]["template"]["name"])

    # Check if method is defined
    if "method" not in config["ssp"]:
        logger.debug("Method not defined, using default method: cubic")
        method = "cubic"
    method = config["ssp"]["method"]

    return ssp.get_lookup(method=method)
