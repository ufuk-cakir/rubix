from rubix.telescope.psf.psf import get_psf_kernel, apply_psf
from rubix.logger import get_logger

from typing import Callable
from .data import RubixData


# TODO: add option to disable PSF convolution
def get_convolve_psf(config: dict) -> Callable:
    """Get the point spread function (PSF) kernel based on the configuration."""
    logger = get_logger(config.get("logger", None))
    # Check if key exists in config file
    if "psf" not in config["telescope"]:
        raise ValueError("PSF configuration not found in telescope configuration")
    if "name" not in config["telescope"]["psf"]:
        raise ValueError("PSF name not found in telescope configuration")

    # Get the PSF kernel based on the configuration
    if config["telescope"]["psf"]["name"] == "gaussian":
        # Check if the PSF size and sigma are defined
        if "size" not in config["telescope"]["psf"]:
            raise ValueError("PSF size not found in telescope configuration")
        if "sigma" not in config["telescope"]["psf"]:
            raise ValueError("PSF sigma not found in telescope configuration")

        m, n = config["telescope"]["psf"]["size"], config["telescope"]["psf"]["size"]
        sigma = config["telescope"]["psf"]["sigma"]
        psf_kernel = get_psf_kernel("gaussian", m, n, sigma=sigma)

    else:
        raise ValueError(
            f"Unknown PSF kernel name: {config['telescope']['psf']['name']}"
        )

    # Define the function to convolve the datacube with the PSF kernel
    def convolve_psf(rubixdata: RubixData) -> RubixData:
        """Convolve the input datacube with the PSF kernel."""
        logger.info("Convolving with PSF...")
        rubixdata.stars.datacube = apply_psf(rubixdata.stars.datacube, psf_kernel)
        return rubixdata

    return convolve_psf
