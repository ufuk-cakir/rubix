"""Save the SSP data to an HDF5 file
adapted from https://github.com/ArgonneCPAC/dsps/blob/main/scripts/write_fsps_data_to_disk.py"""

import argparse
from .fsps_grid import retrieve_ssp_data_from_fsps
import h5py

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("outname", help="Name of the output file")
    args = parser.parse_args()

    ssp_data = retrieve_ssp_data_from_fsps()

    with h5py.File(args.outname, "w") as hdf:
        for key, arr in zip(ssp_data._fields, ssp_data):
            hdf[key] = arr