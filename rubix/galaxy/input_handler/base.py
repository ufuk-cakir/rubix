from abc import ABC, abstractmethod
import os
import h5py
import logging
import astropy.units as u
from typing import List, Union, Optional
from rubix import config
from rubix.logger import get_logger

def create_rubix_galaxy(
    file_path: str,
    particle_data: dict,
    galaxy_data: dict,
    simulation_metadata: dict,
    units: dict,
    config: dict,
    logger: logging.Logger,
):
    logger.debug("Creating Rubix file at path: %s", file_path)

    with h5py.File(file_path, "w") as f:
        meta_group = f.create_group("meta")
        galaxy_group = f.create_group("galaxy")
        particle_group = f.create_group("particles")

        for key, value in simulation_metadata.items():
            meta_group.create_dataset(key, data=value)

        for key, value in galaxy_data.items():
            if isinstance(value, u.Quantity):
                value = value.to(units["galaxy"][key])
            else:
                value = u.Quantity(value, units["galaxy"][key])
            galaxy_group.create_dataset(key, data=value.value)
            galaxy_group[key].attrs["unit"] = str(units["galaxy"][key])

        for key in particle_data:
            particle_group.create_group(key)
            for field, value in particle_data[key].items():
                value = u.Quantity(value, units[key][field]).to(config["particles"][key][field])
                particle_group[key].create_dataset(field, data=value)
                particle_group[key][field].attrs["unit"] = config["particles"][key][field]

    logger.info(f"Rubix file saved at {file_path}")


class BaseHandler(ABC):
    def __init__(self, config, logger_config=None):
        self.config = config
        self._logger = get_logger(logger_config)

    @abstractmethod
    def get_particle_data(self) -> dict:
        pass

    @abstractmethod
    def get_galaxy_data(self) -> dict:
        pass

    @abstractmethod
    def get_simulation_metadata(self) -> dict:
        pass

    @abstractmethod
    def get_units(self) -> dict:
        pass

    def to_rubix(self, output_path: str):
        self._logger.debug("Converting to Rubix format..");

        os.makedirs(output_path, exist_ok=True)

        particle_data = self.get_particle_data()
        galaxy_data = self.get_galaxy_data()
        simulation_metadata = self.get_simulation_metadata()
        units = self.get_units()

        self._check_data(particle_data, galaxy_data, simulation_metadata, units)

        file_path = os.path.join(output_path, "rubix_galaxy.h5")
        self._logger.info(f"Rubix file saved at {file_path}")

        create_rubix_galaxy(
            file_path,
            particle_data,
            galaxy_data,
            simulation_metadata,
            units,
            self.config,
            self._logger,
        )

    def _check_data(self, particle_data, galaxy_data, simulation_metadata, units):
        self._check_galaxy_data(galaxy_data, units)
        self._check_particle_data(particle_data, units)
        self._check_simulation_metadata(simulation_metadata)

    def _check_simulation_metadata(self, simulation_metadata):
        pass

    def _check_galaxy_data(self, galaxy_data, units):
        for field in self.config["galaxy"]:
            if field not in galaxy_data:
                raise ValueError(f"Missing field {field} in galaxy data")
        for field in galaxy_data:
            if field not in units["galaxy"]:
                raise ValueError(f"Units for {field} not found in units")

    def _check_particle_data(self, particle_data, units):
        for key in self.config["particles"]:
            if key not in particle_data:
                raise ValueError(f"Missing particle type {key} in particle data")
            for field in self.config["particles"][key]:
                if field not in particle_data[key]:
                    raise ValueError(f"Missing field {field} in particle data for particle type {key}")

        for key in particle_data:
            for field in particle_data[key]:
                if field not in units[key]:
                    raise ValueError(f"Units for {field} not found in units")
