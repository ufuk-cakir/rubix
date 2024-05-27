from rubix.utils import read_yaml
from rubix.spectra.ssp.grid import SSPGrid, HDF5SSPGrid, pyPipe3DSSPGrid
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
        return HDF5SSPGrid.from_hdf5(config[name], file_location=TEMPLATE_PATH)
    elif config[name]["format"].lower() == "pypipe3d":
        return pyPipe3DSSPGrid.from_fits(config[name], file_location=TEMPLATE_PATH)


    else:
        raise ValueError("Currently only HDF5 format is supported for SSP templates.")

