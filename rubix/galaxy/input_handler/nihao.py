import sys
# Add project path for proper imports
#sys.path.insert(0, 'rubix_path')

from .base import BaseHandler
import pynbody
import numpy as np
from rubix.utils import SFTtoAge
import logging
import astropy.units as u
from rubix.telescope import TelescopeFactory
import matplotlib.pyplot as plt

# Define a custom unit for metallicity (Zsun), if not already present
if not hasattr(u, 'Zsun'):
    u.Zsun = u.def_unit("Zsun", u.dimensionless_unscaled)

class NihaoHandler(BaseHandler):
    def __init__(self, path, halo_path=None, logger=None):
        '''Initialize handler with paths to snapshot and halo files.'''
        super().__init__()
        self.path = path
        self.halo_path = halo_path
        self.logger = logger or self._default_logger()
        self.logger.debug(f"NihaoHandler initialized with path: {path} and halo_path: {halo_path}")
        self.load_data()  # Load data upon initialization
        
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

        # Load halo data if a halo path is provided
        if self.halo_path:
            halos = self.sim.halos(filename=self.halo_path)
            pynbody.analysis.angmom.faceon(halos[1])  # Align along angular momentum
            self.halo_data = halos[1]
            self.logger.info("Halos loaded successfully.")
        else:
            self.halo_data = None
            self.logger.warning("No halo file provided or found.")
        
        # Store essential data for stars, gas, and dark matter in a dictionary
        self.data = {
            'stars': {
                'age': SFTtoAge(self.sim.stars['tform']),
                'mass': self.sim.stars['mass'],
                'metallicity': self.sim.stars['metals'],
                'coords': self.sim.stars['pos'],
                'velocity': self.sim.stars['vel'],
            },
            'gas': {
                'density': self.sim.gas['rho'],
                'temperature': self.sim.gas['temp'],
                'metallicity': self.sim.gas['metals']
            },
            'dm': {
                'mass': self.sim.dm['mass']
            }
        }
        self.logger.info("NIHAO snapshot and halo data loaded successfully.")

    def get_data(self):
        '''Return loaded data for stars, gas, and dark matter.'''
        return self.data

    def get_halo_data(self):
        '''Return halo data if available.'''
        if self.halo_data:
            return {
                'stars': {
                    'mass': self.halo_data.s['mass'],
                    'density': self.halo_data.s['rho']
                },
                'gas': {
                    'density': self.halo_data.g['rho'],
                    'temperature': self.halo_data.g['temp']
                },
                'dm': {
                    'mass': self.halo_data.d['mass']
                }
            }
        else:
            self.logger.warning("No halo data available.")
            return None
            
    def _get_center(self):
        # Calculate the mass-weighted center of stellar particles
        star_positions = self.sim.stars['pos']
        star_masses = self.sim.stars['mass']
        center = np.average(star_positions, weights=star_masses, axis=0)
        return center

    def get_galaxy_data(self):
        '''Return basic galaxy data including center and redshift.'''
        redshift = 0.1  # Placeholder value
        center = self._get_center()  # Compute center from stellar particles
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
        '''Define and return units for all quantities.'''
        return {
            'stars': {
                'mass': u.M_sun,                
                'age': u.Gyr,                   
                'metallicity': u.Zsun,          
                'coords': u.kpc,                
                'velocity': u.km / u.s          
            },
            'gas': {
                'density': u.M_sun / u.kpc**3,  
                'temperature': u.K,             
                'metallicity': u.Zsun           
            },
            'dm': {
                'mass': u.M_sun                 
            },
            'galaxy': {
                'redshift': u.one,              
                'center': u.kpc,                
                'halfmassrad_stars': u.kpc      
            }
        }