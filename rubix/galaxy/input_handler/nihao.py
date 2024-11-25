import sys
from .base import BaseHandler
import pynbody
import numpy as np
from rubix.utils import SFTtoAge
import logging
import astropy.units as u
from rubix.telescope import TelescopeFactory
import matplotlib.pyplot as plt
import yaml
import os 

Zsun = u.def_unit("Zsun", u.dimensionless_unscaled) 

u.add_enabled_units(Zsun) 

# Function to load the YAML configuration
def load_config():
    config_path = "/insert/your/config/path/here"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

class NihaoHandler(BaseHandler):
    def __init__(self, path, halo_path=None, logger=None):
        '''Initialize handler with paths to snapshot and halo files.'''
        config = load_config() 
        super().__init__(config=config)
        self.path = path or config["data_path"]
        self.halo_path = halo_path or config.get("halo_path")
        self.logger = logger or self._default_logger()
        self.load_data()
        
    def _default_logger(self):
        '''Create a default logger if none is provided.'''
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        return logger
    
    def load_data(self):
        '''Load data from snapshot and halo file (if available).'''
        self.sim = pynbody.load(self.path)
        self.sim.physical_units()
        fields = self.config["fields"]

        # Get the particle classes to load from the config or use all by default
        load_classes = self.config.get("load_classes", ["stars", "gas", "dm"])
        self.data = {}
        units = self.get_units()

        # Conditionally load particle classes
        if "stars" in load_classes:
            self.data["stars"] = {
                'age': SFTtoAge(self.sim.stars[fields["stars"]["age"]]) * units["stars"]["age"], 
                'mass': self.sim.stars[fields["stars"]["mass"]] * units["stars"]["mass"], 
                'metallicity': self.sim.stars[fields["stars"]["metallicity"]] * units["stars"]["metallicity"],
                'coords': self.sim.stars[fields["stars"]["coords"]] * units["stars"]["coords"],
                'velocity': self.sim.stars[fields["stars"]["velocity"]] * units["stars"]["velocity"],
            }
        if "gas" in load_classes:
            self.data["gas"] = {
                'density': self.sim.gas[fields["gas"]["density"]] * units["gas"]["density"],
                'temperature': self.sim.gas[fields["gas"]["temperature"]] * units["gas"]["temperature"],
                'metallicity': self.sim.gas[fields["gas"]["metallicity"]] * units["gas"]["metallicity"],
            }
        if "dm" in load_classes:
            self.data["dm"] = {
                'mass': self.sim.dm[fields["dm"]["mass"]] * units["dm"]["mass"],
            }

        self.logger.info(f"NIHAO snapshot and halo data loaded successfully for classes: {load_classes}.")

    def get_data(self):
        '''Return loaded data for stars, gas, and dark matter.'''
        return self.data

    def get_halo_data(self):
        '''Return halo data if available.'''
        if self.halo_path:
            halos = self.sim.halos(filename=self.halo_path)
            pynbody.analysis.angmom.faceon(halos[1])
            self.halo_data = halos[1]
            self.logger.info("Halos loaded successfully.")
        else:
            self.logger.warning("No halo file provided or found.")
            self.halo_data = None
        return self.halo_data
            
    def _get_center(self):
        '''Calculate the center of the galaxy based on stellar positions and masses.'''
        star_positions = self.sim.stars['pos']
        star_masses = self.sim.stars['mass']

        center = np.average(star_positions, weights=star_masses, axis=0)
        
        # Get units from the configuration using the existing get_units method
        units = self.get_units()
        center_unit = units["galaxy"]["center"]
        
        # Convert center to a Quantity with the correct unit
        center_with_unit = u.Quantity(center, center_unit)
        return center_with_unit

    def get_galaxy_data(self):
        '''Return basic galaxy data including center and redshift.'''
        redshift = self.config.get("galaxy", {}).get("redshift", 0.1)  
        
        center = self._get_center()  
        
        halfmassrad_stars = self.config["galaxy"]["halfmassrad_stars"]
        
        data = {
            "redshift": redshift,
            "center": center,
            "halfmassrad_stars": halfmassrad_stars,
        }
        return data
    def get_particle_data(self):
        '''Return particle data dictionary.'''
        return self.data

    def get_simulation_metadata(self):
        '''Return metadata for the simulation.'''
        return {
            'path': self.path,
            'halo_path': self.halo_path,
            'logger': str(self.logger)
        }

    def get_units(self):
        '''Define and return units for all quantities based on the YAML config.'''
        units = self.config["units"]
        
        # Convert units from strings to astropy.units using a mapping dictionary
        unit_map = {
            "Msun": u.M_sun,
            "Gyr": u.Gyr,
            "Zsun": u.Unit("Zsun"),
            "kpc": u.kpc,
            "km/s": u.km / u.s,
            "Msun/kpc^3": u.M_sun / u.kpc**3,
            "K": u.K,
            "dimensionless": u.dimensionless_unscaled
}
        
        # Use unit_map to convert units from YAML config to astropy units
        parsed_units = {
            'stars': {key: unit_map[unit] for key, unit in units["stars"].items()},
            'gas': {key: unit_map[unit] for key, unit in units["gas"].items()},
            'dm': {key: unit_map[unit] for key, unit in units["dm"].items()},
            'galaxy': {key: unit_map[unit] for key, unit in units["galaxy"].items()}
        }
        
        return parsed_units
