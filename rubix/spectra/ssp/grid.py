from jaxtyping import Float, Array
import equinox as eqx
from astropy import units as u
import os
import h5py
import jax.numpy as jnp
from rubix import config as rubix_config
from typing import Dict

SSP_UNITS = rubix_config["ssp"]["units"]


class SSPGrid(eqx.Module):
    """
    Base class for all SSP
    """

    age: Float[Array, " age_bins"]
    metallicity: Float[Array, " metallicity_bins"]
    wavelength: Float[Array, " wavelength_bins"]
    flux: Float[Array, "age_bins metallicity_bins wavelength_bins"]
    units: Dict[str, str] = eqx.field(default_factory=dict)

    def __init__(self, age, metallicity, wavelength, flux):
        self.age = jnp.asarray(age)
        self.metallicity = jnp.asarray(metallicity)
        self.wavelength = jnp.asarray(wavelength)
        self.flux = jnp.asarray(flux)
        self.units = SSP_UNITS

    @staticmethod
    def convert_units(data, from_units, to_units):
        quantity = u.Quantity(data, from_units)
        return quantity.to(to_units).value

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
        if config.get("format", "").lower() != "hdf5":
            raise ValueError("Configured file format is not HDF5.")

        file_path = os.path.join(file_location, config["file_name"])
        ssp_data = {}
        with h5py.File(file_path, "r") as f:
            for field_name, field_info in config["fields"].items():
                data = f[field_info["name"]][:]  # type: ignore
                data = jnp.power(10, data) if field_info["in_log"] else data  # type: ignore
                data = cls.convert_units(
                    data, field_info["units"], SSP_UNITS[field_name]
                )
                ssp_data[field_name] = data

        grid = cls(**ssp_data)
        grid.__class__.__name__ = config["name"]
        return grid
