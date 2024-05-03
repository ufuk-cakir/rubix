from jaxtyping import Float, Array
from equinox import Module, field  # type: ignore #noqa
from astropy import units as u
import os
import h5py

# TODO move this to the config file
SSP_UNITS = {
    "age": "Gyr",
    "metallicity": "",
    "wavelength": "Angstrom",
    "flux": "Lsun/Angstrom",
}


class SSPGrid(Module):
    """
    Base class for all SSP
    """

    # TODO: convert this into jax arrays
    age: Float[Array, " age_bins"]
    metallicity: Float[Array, " metallicity_bins"]
    wavelength: Float[Array, " wavelength_bins"]
    flux: Float[Array, " age_bins metallicity_bins wavelength_bins"]

    @classmethod
    def from_hdf5(cls, config: dict, file_location: str) -> "SSPGrid":
        """
        Load a SSP grid from a HDF5 file.

        Parameters
        ----------
        config : dict
            Configuration dictionary.

        Returns
        -------
        SSPGrid
            The SSP grid in the correct units.
        """
        if config["format"].lower() != "hdf5":
            raise ValueError("The format of the file is not HDF5.")

        file_path = os.path.join(file_location, config["file_name"])
        fields_dict = config["fields"]
        ssp_data = {}
        with h5py.File(file_path, "r") as f:
            for field_name, data in fields_dict.items():
                field_name_in_file = data["name"]
                field_units = data["units"]
                field_log = data["in_log"]

                if field_log:
                    field_data = 10 ** f[field_name_in_file][:]  # type: ignore

                # Get the data in the correct units using astropy
                field_data = u.Quantity(f[field_name_in_file][:], field_units)  # type: ignore

                # Convert to the correct units
                field_data = field_data.to(SSP_UNITS[field_name])
                ssp_data[field_name] = field_data

        grid = cls(**ssp_data)
        grid.__class__.__name__ = config["name"]
        return grid
