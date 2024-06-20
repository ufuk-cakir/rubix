from .base import BaseHandler  # type: ignore
import os
import h5py
import numpy as np
from rubix.utils import convert_values_to_physical, SFTtoAge
from rubix import config


class IllustrisHandler(BaseHandler):
    MAPPED_FIELDS = config["IllustrisHandler"]["MAPPED_FIELDS"]
    # This Dictionary maps the particle name in the simulation to the name used in Rubix
    MAPPED_PARTICLE_KEYS = config["IllustrisHandler"]["MAPPED_PARTICLE_KEYS"]

    # This dictiony map the keys of the simulation metadata to the keys used in Rubix
    # This also defines the required fields for the simulation metadata, which are used to check if the file is valid
    SIMULATION_META_KEYS = config["IllustrisHandler"]["SIMULATION_META_KEYS"]

    GALAXY_SUBHALO_KEYS = config["IllustrisHandler"]["GALAXY_SUBHALO_KEYS"]

    # This dictionary defines the units we get from the simulation
    UNITS = config["IllustrisHandler"]["UNITS"]

    ILLUSTRIS_DATA = config["IllustrisHandler"]["ILLUSTRIS_DATA"]

    def __init__(self, path, logger=None):
        super().__init__()

        if logger is not None:
            self._logger = logger
        else:
            import logging

            self._logger = logging.getLogger(__name__)
            self._logger.setLevel(logging.INFO)

        self.path = path
        # Check if paths are valid
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"File {self.path} not found")
        (
            self.simulation_metadata,
            self.particle_data,
            self.galaxy_data,
        ) = self._load_data()

    def get_particle_data(self):
        return self.particle_data

    def get_galaxy_data(self):
        return self.galaxy_data

    def get_simulation_metadata(self):
        return self.simulation_metadata

    def get_units(self):
        return self.UNITS
    
    def _check_fields(self, f):
        self._logger.debug("Checking if the fields are present in the file...")
        self._logger.debug(f"Keys in the file: {f.keys()}")
        self._logger.debug(f"Expected fields: {self.ILLUSTRIS_DATA}")

        present_fields = set(f.keys())
        expected_fields = set(self.ILLUSTRIS_DATA)

        matching_fields = present_fields.intersection(expected_fields)
        extra_fields = present_fields - expected_fields

        if not matching_fields:
            raise ValueError(f"No expected fields found in the file. Expected at least one of: {self.ILLUSTRIS_DATA}")
        
        if extra_fields:
            raise ValueError(f"Unexpected fields found in the file: {extra_fields}")

        self._logger.debug(f"Matching fields: {matching_fields}")

    def _load_data(self):
        # open the file
        with h5py.File(self.path, "r") as f:
            self._logger.debug(
                f"Loading data from Illustris file located in {self.path}.."
            )
            # Check if the file has the required fields
            #self._check_fields(f)
            # Get information from the header

            # TIME is the scale factor of the simulation
            # HUBBLE_PARAM is the Hubble parameter
            # these values are used to convert the values to physical units
            # PARTICLE_KEYS are the keys of the particle types in the file

            (
                self.TIME,
                self.HUBBLE_PARAM,
                self.PARTICLE_KEYS,
            ) = self._get_data_from_header(f)
            # Get simulation metadata
            simulation_metadata = self._get_metadata(f)

            # Get the data of the different particle types
            particle_data = self._get_data(f)

            # Get the Subhalo Galaxy Data
            galaxy_data = self._get_galaxy_data(f)

        return simulation_metadata, particle_data, galaxy_data

    def _get_data_from_header(self, f):
        # Check if the file has the required fields
        if "Time" not in f["Header"].attrs:
            raise ValueError("Time not found in the header attributes")
        if "HubbleParam" not in f["Header"].attrs:
            raise ValueError("HubbleParam not found in the header attributes")
        # Get the values
        TIME = f["Header"].attrs["Time"]
        HUBBLE_PARAM = f["Header"].attrs["HubbleParam"]
        PARTICLE_KEYS = self._get_particle_keys(f)
        return TIME, HUBBLE_PARAM, PARTICLE_KEYS

    def _get_particle_keys(self, f):
        # Check if the keys are supported
        keys = [key for key in f.keys() if key.startswith("PartType")]

        supported_keys = self.MAPPED_PARTICLE_KEYS.keys()
        for key in keys:
            if key not in supported_keys:
                raise NotImplementedError(
                    f"{key} is not supported. Currently only {supported_keys} are supported"
                )
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
        present_fields = set(f.keys())
        attributes_coords = None

        for part_type in present_fields:
                attributes_coords = f[part_type]["Coordinates"].attrs
                break  # Stop after finding the first match


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
        present_fields = set(f.keys())
        attributes_coords = None

        for part_type in present_fields:
                attributes_coords = f[part_type]["Coordinates"].attrs
                break  # Stop after finding the first match
        
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
            # Get the particle data
            data_particle = self._get_particle_data(f, part_type)

            # Save with the correct key
            data[self.MAPPED_PARTICLE_KEYS[part_type]] = data_particle
        return data

    def _get_metadata(self, f):
        data = {}
        for keys in f["Header"].attrs:
            data[keys] = f["Header"].attrs[keys]
        return data

    def _get_particle_data(self, f, part_type):
        """Convert values to physical units"""
        # self._logger.debug(
        #    f"Calculating {part_type} particles parameters in physical units.."
        # )
        part_data = {}
        for key in f[part_type].keys():
            # Check if key is supported, if not, raise a warning and skip
            if key not in self.MAPPED_FIELDS[part_type]:
                raise NotImplementedError(
                    f"{key} is not supported. Currently only {self.MAPPED_FIELDS[part_type].keys()} are supported"
                )

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
                self._logger.debug("Converting Stellar Formation Time to Age")
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
