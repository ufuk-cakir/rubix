from rubix.spectra.ssp.grid import SSPGrid, HDF5SSPGrid, pyPipe3DSSPGrid
from rubix.spectra.ssp.fsps_grid import retrieve_ssp_data_from_fsps
from rubix import config as rubix_config
from rubix.paths import TEMPLATE_PATH


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
        return HDF5SSPGrid.from_file(config[name], file_location=TEMPLATE_PATH)
    elif config[name]["format"].lower() == "pypipe3d":
        return pyPipe3DSSPGrid.from_file(config[name], file_location=TEMPLATE_PATH)
    elif config[name]["format"].lower() == "fsps":
        return retrieve_ssp_data_from_fsps()
    else:
        raise ValueError("Currently only HDF5 format and fits files in the format of pyPipe3D format are supported for SSP templates.")

