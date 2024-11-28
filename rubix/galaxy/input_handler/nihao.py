from .base import BaseHandler
import pynbody
import numpy as np
from rubix.utils import SFTtoAge
import logging
import astropy.units as u
import yaml
import os

Zsun = u.def_unit("Zsun", u.dimensionless_unscaled)
u.add_enabled_units(Zsun)

class NihaoHandler(BaseHandler):
    def __init__(self, path, halo_path=None, logger=None, config=None):
        """Initialize handler with paths to snapshot and halo files."""
        self.path = path
        self.halo_path = halo_path
        self.nihao_config = config or self._load_config()
        self.logger = logger or self._default_logger()
        super().__init__()
        self.load_data()

    def _load_config(self):
        """Load the YAML configuration."""
        config_path = "/insert/your/config/path/here"
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with open(config_path, "r") as file:
            return yaml.safe_load(file)

    def _default_logger(self):
        """Create a default logger if none is provided."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        return logger

    def load_data(self):
        """Load data from snapshot and halo file (if available)."""
        self.sim = pynbody.load(self.path)
        self.sim.physical_units()
        fields = self.nihao_config["fields"]

        # Get the particle classes to load from the config or use all by default
        load_classes = self.nihao_config.get("load_classes", ["stars", "gas", "dm"])
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
        """Return loaded data for stars, gas, and dark matter."""
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

    def get_galaxy_data(self):
        """Return basic galaxy data including center and redshift."""
        redshift = self.nihao_config.get("galaxy", {}).get("redshift", 0.1)
        center = self._get_center()
        halfmassrad_stars = self.nihao_config["galaxy"]["halfmassrad_stars"]
        return {
            "redshift": redshift,
            "center": center,
            "halfmassrad_stars": halfmassrad_stars,
        }

    def _get_center(self):
        """Calculate the center of the galaxy based on stellar positions and masses."""
        star_positions = self.sim.stars['pos']
        star_masses = self.sim.stars['mass']
        center = np.average(star_positions, weights=star_masses, axis=0)

        # Convert center to astropy Quantity
        units = self.get_units()
        center_unit = units["galaxy"]["center"]
        return u.Quantity(center, center_unit)

    def get_particle_data(self):
        """Return particle data dictionary."""
        return self.data

    def get_simulation_metadata(self):
        """Return metadata for the simulation."""
        return {
            'path': self.path,
            'halo_path': self.halo_path,
            'logger': str(self.logger),
        }

    def get_units(self):
        """Define and return units for all quantities based on the YAML config."""
        unit_map = {
            "Msun": u.M_sun,
            "Gyr": u.Gyr,
            "Zsun": u.Unit("Zsun"),
            "kpc": u.kpc,
            "km/s": u.km / u.s,
            "Msun/kpc^3": u.M_sun / u.kpc**3,
            "K": u.K,
            "dimensionless": u.dimensionless_unscaled,
        }
        units = self.nihao_config["units"]
        return {
            category: {field: unit_map[unit] for field, unit in units[category].items()}
            for category in units
        }
