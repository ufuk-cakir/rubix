import sys
from .base import BaseHandler
import pynbody
import numpy as np
from rubix.utils import SFTtoAge
import logging
import astropy.units as u
from rubix.telescope import TelescopeFactory
import matplotlib.pyplot as plt


# Function to load the YAML configuration
def load_config():
    with open("config/nihao_config.yml", "r") as file:
        config = yaml.safe_load(file)
    return config

class NihaoHandler(BaseHandler):
    def __init__(self, path, halo_path=None, logger=None):
        '''Initialize handler with paths to snapshot and halo files.'''
        super().__init__()
        self.config = load_config()
        self.path = path or self.config["data_path"]
        self.halo_path = halo_path or self.config.get("halo_path")
        self.logger = logger or self._default_logger()
        self.logger.debug(f"NihaoHandler initialized with path: {path} and halo_path: {halo_path}")
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
        self.logger.debug("Available star attributes in sim.stars: %s", self.sim.stars.loadable_keys())
        
        # Store data for stars, gas, and dark matter in a dictionary
        self.data = {
            'stars': {
                'age': SFTtoAge(self.sim.stars[fields["stars"]["age"]]),
                'mass': self.sim.stars[fields["stars"]["mass"]],
                'metallicity': self.sim.stars[fields["stars"]["metallicity"]],
                'coords': self.sim.stars[fields["stars"]["coords"]],
                'velocity': self.sim.stars[fields["stars"]["velocity"]],
            },
            'gas': {
                'density': self.sim.gas[fields["gas"]["density"]],
                'temperature': self.sim.gas[fields["gas"]["temperature"]],
                'metallicity': self.sim.gas[fields["gas"]["metallicity"]]
            },
            'dm': {
                'mass': self.sim.dm[fields["dm"]["mass"]]
            }
        }
        self.logger.info("NIHAO snapshot and halo data loaded successfully.")

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
        # Calculate the mass-weighted center of stellar particles
        star_positions = self.sim.stars['pos']
        star_masses = self.sim.stars['mass']
        center = np.average(star_positions, weights=star_masses, axis=0)
        return center

    def get_galaxy_data(self):
        '''Return basic galaxy data including center and redshift.'''
        redshift = self.config.get("galaxy", {}).get("dist_z", 0.1)  
        center = self._get_center()  # Calculate center from stellar particles
        halfmassrad_stars = 5.0  # Placeholder for half-mass radius

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
            "Zsun": u.def_unit("Zsun", u.dimensionless_unscaled),
            "kpc": u.kpc,
            "km/s": u.km / u.s,
            "Msun/kpc^3": u.M_sun / u.kpc**3,
            "K": u.K
        }
        
        # Use unit_map to convert units from YAML config to astropy units
        parsed_units = {
            'stars': {key: unit_map[unit] for key, unit in units["stars"].items()},
            'gas': {key: unit_map[unit] for key, unit in units["gas"].items()},
            'dm': {key: unit_map[unit] for key, unit in units["dm"].items()}
        }
        
        return parsed_units
