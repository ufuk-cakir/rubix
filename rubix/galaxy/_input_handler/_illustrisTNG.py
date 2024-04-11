from ._input_handler import InputHandler
import os
import h5py
import numpy as np
import warnings
from rubix.logger import logger
from rubix.utils import convert_values_to_physical, SFTtoAge


class IllustrisHandler(InputHandler):

    # TODO [RBX-18] Change names to be more descriptive
    MAPPED_FIELDS = {
        "PartType4": {
            "Coordinates": "coords",
            "Masses": "mass",
            "GFM_Metallicity": "metallicity",
            "Velocities": "velocity",
            "GFM_StellarFormationTime": "age",  # for this we convert SFT to age
        }
    }
    MAPPED_PARTICLE_KEYS = {
        "PartType4": "stars",
        # Currently only PartType4 is supported
    }

    SIMULATION_META_KEYS = {
        "name": "SimulationName",
        "snapshot": "SnapshotNumber",
        "redshift": "Redshift",
        "subhalo_id": "CutoutID",
        "api_request": "CutoutRequest",
    }

    GALAXY_SUBHALO_KEYS = {"halfmassrad_stars": "halfmassrad_stars"}

    UNITS = {
        "stars": {
            "coords": "cm",
            "mass": "g",
            "metallicity": "dimensionless",
            "velocity": "cm/s",
            "age": "Gyr",
        },
        "galaxy": {
            "center": "cm",
            "halfmassrad_stars": "cm",
            "redshift": "dimensionless",
        },
    }

    def __init__(self, path, output_path):
        self.path = path
        self.output_path = output_path

        # Check if paths are valid
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"File {self.path} not found")
        if not os.path.exists(self.output_path):
            raise FileNotFoundError(f"Output path {self.output_path} not found")
        self.simulation_metadata, self.particle_data, self.galaxy_data = (
            self._load_data()
        )

    def get_particle_data(self):
        return self.particle_data

    def get_galaxy_data(self):
        return self.galaxy_data

    def get_simulation_metadata(self):
        return self.simulation_metadata

    def get_units(self):
        return self.UNITS

    def _load_data(self):
        # open the file
        with h5py.File(self.path, "r") as f:

            # Get information from the header
            # TIME is the scale factor of the simulation
            # HUBBLE_PARAM is the Hubble parameter
            # these values are used to convert the values to physical units
            # PARTICLE_KEYS are the keys of the particle types in the file
            self.TIME = f["Header"].attrs["Time"]
            self.HUBBLE_PARAM = f["Header"].attrs["HubbleParam"]
            self.PARTICLE_KEYS = self._get_particle_keys(f)

            # Get simulation metadata
            simulation_metadata = self._get_metadata(f)

            # Get the data of the different particle types
            particle_data = self._get_data(f)

            # Get the Subhalo Galaxy Data
            galaxy_data = self._get_galaxy_data(f)

        return simulation_metadata, particle_data, galaxy_data

    def _get_particle_keys(self, f):
        keys = [key for key in f.keys() if "PartType" in key]
        # check if there are any particle types
        if len(keys) == 0:
            raise ValueError("No particle types found in the file")
        return keys

    def _get_galaxy_data(self, f):

        redshift = f["Header"].attrs["Redshift"]
        center = self._get_center(f)
        halfmassrad_stars = self._get_halfmassrad_stars(f)
        data = {
            "redshift": redshift,
            "center": center,
            "halfmassrad_stars": halfmassrad_stars,
        }
        return data

    def _get_halfmassrad_stars(self, f):
        halfmass_rad_stars = f["SubhaloData"]["halfmassrad_stars"][()]
        # Get the attributes to convert values from Coordinates field
        attributes_coords = f["PartType4"]["Coordinates"].attrs
        # Convert to physical Units
        halfmass_rad_stars = convert_values_to_physical(
            halfmass_rad_stars,
            self.TIME,
            attributes_coords["a_scaling"],
            self.HUBBLE_PARAM,
            attributes_coords["h_scaling"],
            attributes_coords["to_cgs"],
        )
        return halfmass_rad_stars

    def _get_center(self, f):
        pos_x = f["SubhaloData"]["pos_x"][()]
        pos_y = f["SubhaloData"]["pos_y"][()]
        pos_z = f["SubhaloData"]["pos_z"][()]

        center = np.array([pos_x, pos_y, pos_z])

        # Get the attributes to convert values from Coordinates field
        attributes_coords = f["PartType4"]["Coordinates"].attrs
        # Convert to physical Units
        center = convert_values_to_physical(
            center,
            self.TIME,
            attributes_coords["a_scaling"],
            self.HUBBLE_PARAM,
            attributes_coords["h_scaling"],
            attributes_coords["to_cgs"],
        )

        return center

    def _get_data(self, f):
        data = {}
        for part_type in self.PARTICLE_KEYS:
            # Check if the particle type is supported
            if part_type not in self.MAPPED_PARTICLE_KEYS:
                # Raise warning and skip
                warnings.warn(
                    f"{part_type} is currently not supported. Currently only {self.MAPPED_PARTICLE_KEYS.keys()} are supported. Skipping.."
                )
                continue
                # TODO or do we want to raise an error?
                # raise NotImplementedError(
                #    f"{part_type} is not supported. Currently only {self.MAPPED_PARTICLE_KEYS.keys()} are supported"
                # )
            # Get the particle data
            data_particle = self._get_particle_data(f, part_type)

            # Save with the correct key
            data[self.MAPPED_PARTICLE_KEYS[part_type]] = data_particle
        return data

    def _get_metadata(self, f):
        data = {}
        for key in self.SIMULATION_META_KEYS:
            # Check if the key is in the file
            if self.SIMULATION_META_KEYS[key] not in f["Header"].attrs:
                raise ValueError(
                    f"{self.SIMULATION_META_KEYS[key]} not found in the simulation metadata"
                )
            data[key] = f["Header"].attrs[self.SIMULATION_META_KEYS[key]]

        return data

    def _get_particle_data(self, f, part_type):
        """Convert values to physical units"""
        # self._logger.debug(
        #    f"Calculating {part_type} particles parameters in physical units.."
        # )

        # Check if part_type is supported
        if part_type not in self.MAPPED_FIELDS:
            # This should not happen, as we already checked for the keys. If it does, raise an error
            raise ValueError(
                f"{part_type} is not supported. Currently only {self.MAPPED_FIELDS.keys()} are supported"
            )

        part_data = {}
        for key in f[part_type].keys():
            # Check if key is supported, if not, raise a warning and skip
            if key not in self.MAPPED_FIELDS[part_type]:
                warnings.warn(f"{key} is not supported. Skipping..")
                continue

            values = f[part_type][key][()]
            attributes = f[part_type][key].attrs
            physical_values = convert_values_to_physical(
                values,
                self.TIME,
                attributes["a_scaling"],
                self.HUBBLE_PARAM,
                attributes["h_scaling"],
                attributes["to_cgs"],
            )
            # Look for the correct field name

            if key == "GFM_StellarFormationTime":
                logger.debug("Converting Stellar Formation Time to Age")
                physical_values = self._convert_stellar_formation_time(physical_values)

            key = self.MAPPED_FIELDS[part_type][key]
            part_data[key] = physical_values

        # Check if all fields are present
        if len(part_data) != len(self.MAPPED_FIELDS[part_type]):
            raise ValueError(
                f"Missing fields in particle data. Expected: {self.MAPPED_FIELDS[part_type].keys()}, got: {part_data.keys()}"
            )
        return part_data

    def _convert_stellar_formation_time(self, sft):
        return SFTtoAge(sft)  # in Gyr
