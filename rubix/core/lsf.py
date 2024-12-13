from rubix.telescope.lsf.lsf import apply_lsf
from .telescope import get_telescope
from typing import Callable
from rubix.logger import get_logger
from .data import RubixData


def get_convolve_lsf(config: dict) -> Callable:
    """Get the function to convolve with the Line Spread Function (LSF) based on the configuration."""
    logger = get_logger(config.get("logger", None))
    # Check if key exists in config file
    if "lsf" not in config["telescope"]:
        raise ValueError("LSF configuration not found in telescope configuration")

    if "sigma" not in config["telescope"]["lsf"]:
        raise ValueError("LSF sigma size not found in telescope configuration")

    sigma = config["telescope"]["lsf"]["sigma"]

    telescope = get_telescope(config)

    wave_resolution = telescope.wave_res  # Wave Relolution of the telescope

    # Define the function to convolve the datacube with the PSF kernel
    def convolve_lsf(rubixdata: RubixData) -> RubixData:
        """Convolve the input datacube with the LSF."""
        logger.info("Convolving with LSF...")
        rubixdata.stars.datacube = apply_lsf(
            datacube=rubixdata.stars.datacube,
            lsf_sigma=sigma,
            wave_resolution=wave_resolution,
        )
        return rubixdata

    return convolve_lsf
