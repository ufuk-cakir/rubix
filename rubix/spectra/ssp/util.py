"""Save the SSP data to an HDF5 file
adapted from https://github.com/ArgonneCPAC/dsps/blob/main/scripts/write_fsps_data_to_disk.py"""

from .fsps_grid import retrieve_ssp_data_from_fsps
from rubix.paths import TEMPLATE_PATH
import h5py, os

def write_fsps_data_to_disk(outname: str, file_location = TEMPLATE_PATH):
    """
    Write FSPS ssp templagte data to disk.

    Args:
        outname (str): The name of the output file.
        file_location (str, optional): The location where the file will be saved. Defaults to TEMPLATE_PATH.

    Returns:
        None
    """

    ssp_data = retrieve_ssp_data_from_fsps()
    file_path = os.path.join(file_location, outname)

    with h5py.File(file_path, "w") as hdf:
        for key, arr in zip(ssp_data.keys(), ssp_data):
            hdf[key] = arr