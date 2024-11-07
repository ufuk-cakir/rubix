from .base import BaseHandler
import pynbody
import numpy as np
from rubix.utils import SFTtoAge
import logging

class NihaoHandler(BaseHandler):
    def __init__(self, path, halo_path=None, logger=None):
        super().__init__()
        self.path = path
        self.halo_path = halo_path
        self.logger = logger or self._default_logger()

    def _default_logger(self):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        return logger

    def load_data(self):
        sim = pynbody.load(self.path)
        sim.physical_units()
        
        if self.halo_path:
            halos = sim.halos(filename=self.halo_path)
            pynbody.analysis.angmom.faceon(halos[1])
            self.halo_data = halos[1] 
            self.logger.info("Halos loaded successfully.")
        else:
            self.halo_data = None
            self.logger.warning("No halo file provided or found.")

        self.data = {
            'stars': {
                'age': SFTtoAge(sim.stars['tform']),
                'mass': sim.stars['mass'],
                'metallicity': sim.stars['metals']
            },
            'gas': {
                'density': sim.gas['rho'],
                'temperature': sim.gas['temp'],
                'metallicity': sim.gas['metals']
            },
            'dm': {
                'mass': sim.dm['mass']
            }
        }

        self.logger.info("NIHAO snapshot and halo data loaded successfully.")

    def get_data(self):
        return self.data

    def get_halo_data(self):
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

    #placeholder
    def get_galaxy_data(self):
        return self.data.get('stars')

    def get_particle_data(self):
        return self.data

    def get_simulation_metadata(self):
        return {
            'path': self.path,
            'halo_path': self.halo_path,
            'logger': str(self.logger)
        }

    def get_units(self):
        return {
            'mass': 'Msol',
            'density': 'Msol/kpc^3',
            'temperature': 'K',
            'metallicity': 'Zsun'
        }
