from abc import ABC, abstractmethod
import os
import h5py
from rubix.logger import logger


class InputHandler(ABC):
    SUPPORTED_UNITS = ["cm", "g", "dimensionless", "cm/s", "Gyr"]

    REQUIRED_GALAXY_FIELDS = ["redshift", "center", "halfmassrad_stars"]
    REQUIRED_PARTICLE_FIELDS = {
        "stars": ["coords", "mass", "metallicity", "velocity", "age"]
    }

    def __init__(self, output_path):
        self.output_path = output_path

    @abstractmethod
    def get_particle_data(self) -> dict:
        """Returns the particle data in the required format"""

    @abstractmethod
    def get_galaxy_data(self) -> dict:
        """Returns the galaxy data in the required format"""

    @abstractmethod
    def get_simulation_metadata(self) -> dict:
        """Returns the simulation meta data in the required format"""

    @abstractmethod
    def get_units(self) -> dict:
        """Returns the units in the required format"""

    def convert_to_rubix(self):
        logger.debug("Converting to Rubix format..")

        # Get the data
        particle_data = self.get_particle_data()
        galaxy_data = self.get_galaxy_data()
        simulation_metadata = self.get_simulation_metadata()

        # Get the units
        units = self.get_units()

        # Check if the input data is valid and in the correct format
        self._check_data(particle_data, galaxy_data, simulation_metadata, units)

        # Create the Rubix h5 file
        file_path = os.path.join(self.output_path, "rubix_galaxy.h5")
        with h5py.File(file_path, "w") as f:
            # Create groups
            meta_group = f.create_group("meta")
            galaxy_group = f.create_group("galaxy")
            particle_group = f.create_group("particles")

            # Save the simulation metadata as
            for key, value in simulation_metadata.items():
                meta_group.create_dataset(key, data=value)

            # Save the galaxy data: Create a dataset for each field and add the units as attributes
            for key, value in galaxy_data.items():
                galaxy_group.create_dataset(key, data=value)
                galaxy_group[key].attrs["unit"] = units["galaxy"][key]

            # Save the particle data: Create a dataset for each field and add the units as attributes
            for key in particle_data:
                particle_group.create_group(key)
                for field, value in particle_data[key].items():
                    particle_group[key].create_dataset(field, data=value)  # ignore:
                    particle_group[key][field].attrs["unit"] = units[key][field]

        logger.debug(f"Rubix file saved at {file_path}")

    def _check_data(self, particle_data, galaxy_data, simulation_metadata, units):
        # Check if all required fields are present
        self._check_galaxy_data(galaxy_data, units)
        self._check_particle_data(particle_data, units)
        self._check_simulation_metadata(simulation_metadata)

    def _check_simulation_metadata(self, simulation_metadata):
        """Check if all required fields are present in the simulation metadata

        Currently we do not have any required fields to check here.
        """

    def _check_galaxy_data(self, galaxy_data, units):
        # Check if all required fields are present
        for field in self.REQUIRED_GALAXY_FIELDS:
            if field not in galaxy_data:
                raise ValueError(f"Missing field {field} in galaxy data")
        # Check if the units are correct
        for field in galaxy_data:
            if field not in units["galaxy"]:
                raise ValueError(f"Units for {field} not found in units")
            if units["galaxy"][field] not in self.SUPPORTED_UNITS:
                raise ValueError(f"Units for {field} not supported")

    def _check_particle_data(self, particle_data, units):
        # Check if all required fields are present
        for key in self.REQUIRED_PARTICLE_FIELDS:
            if key not in particle_data:
                raise ValueError(f"Missing particle type {key} in particle data")
            for field in self.REQUIRED_PARTICLE_FIELDS[key]:
                if field not in particle_data[key]:
                    raise ValueError(
                        f"Missing field {field} in particle data for particle type {key}"
                    )

        # Check if the units are correct
        for key in particle_data:
            for field in particle_data[key]:
                if field not in units[key]:
                    raise ValueError(f"Units for {field} not found in units")
                if units[key][field] not in self.SUPPORTED_UNITS:
                    raise ValueError(f"Units for {field} not supported")
