from rubix.spectra.ssp.grid import SSPGrid, HDF5SSPGrid, pyPipe3DSSPGrid
from rubix.spectra.ssp.fsps_grid import write_fsps_data_to_disk
from rubix import config as rubix_config
from rubix.paths import TEMPLATE_PATH
from rubix.logger import get_logger


def get_ssp_template(name: str) -> SSPGrid:
    """
    Get the SSP template from the configuration file.

    Returns
    -------
    SSPGrid
        The SSP template.
    """

    config = rubix_config["ssp"]["templates"]

    # Setup a logger based on the config
    logger = get_logger()

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
        if config[name]["source"] == "load_from_file":
            return HDF5SSPGrid.from_file(config[name], file_location=TEMPLATE_PATH)
        elif config[name]["source"] == "rerun_from_scratch":
            logger.info(
                "Running fsps to generate SSP templates. This may take a while."
            )
            write_fsps_data_to_disk(
                config[name]["file_name"], file_location=TEMPLATE_PATH
            )
            return HDF5SSPGrid.from_file(config[name], file_location=TEMPLATE_PATH)
        else:
            raise ValueError(
                f"The source {config[name]['source']} of the FSPS SSP template is not supported."
            )
    else:
        raise ValueError(
            "Currently only HDF5 format and fits files in the format of pyPipe3D format are supported for SSP templates."
        )
