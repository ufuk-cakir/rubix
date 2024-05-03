from rubix.utils import read_yaml
from rubix.spectra.ssp.grid import SSPGrid
from rubix import config as rubix_config
from rubix.paths import TEMPLATE_PATH
import os


def get_ssp_template(name: str) -> SSPGrid:
    """
    Get the SSP template from the configuration file.

    Returns
    -------
    SSPGrid
        The SSP template.
    """

    config = rubix_config["ssp"]["templates"]
    # Check if the template exists in config
    if name not in config:
        raise ValueError(
            f"SSP template {name} not found in the supported configuration file."
        )

    if config[name]["format"].lower() == "hdf5":
        try:
            return SSPGrid.from_hdf5(config[name], file_location=TEMPLATE_PATH)
        except ValueError as e:
            raise ValueError(f"Error loading SSP template {name}: {e}")

    else:
        raise ValueError("Currently only HDF5 format is supported for SSP templates.")

