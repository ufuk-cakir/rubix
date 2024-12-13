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
        if "particles" not in self.config:
            self.config["particles"] = {}

        if "gas" not in self.config["particles"]:
            self.config["particles"]["gas"] = {}

        if "temperature" not in self.config["particles"]["gas"]:
            self.config["particles"]["gas"]["temperature"] = "K"

        if "dm" not in self.config["particles"]:
            self.config["particles"]["dm"] = {}

        if "mass" not in self.config["particles"]["dm"]:
            self.config["particles"]["dm"]["mass"] = "Msun"
        self.load_data()

    def _load_config(self):
        """Load the YAML configuration."""
        config_path = os.path.join(
            os.path.dirname(__file__),
            "../../config/nihao_config.yml" 
        )
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

        load_classes = self.nihao_config.get("load_classes", ["stars", "gas", "dm"])
        self.data = {}
        units = self.get_units()

        # Center the galaxy based on halo data if available
        self.get_halo_data()
        if self.halo_data:
            halo_positions = self.halo_data['pos'].view(np.ndarray)
            halo_masses = self.halo_data['mass'].view(np.ndarray)
            halo_center = (
                np.average(halo_positions, axis=0, weights=halo_masses)
                if halo_masses.size == halo_positions.shape[0]
                else halo_positions.mean(axis=0)
            )
            self.sim['pos'] -= halo_center
            self.center = u.Quantity([0, 0, 0], self.get_units()["galaxy"]["center"])
        else:
            self.logger.warning("No halo data available. Skipping centering.")

        # Load data for stars, gas, and dark matter
        for cls in load_classes:
            if cls == "stars":
                self.data["stars"] = {
                    field: self.sim.stars.get(fields["stars"].get(field, ""), np.zeros(len(self.sim.stars))) * units["stars"].get(field, u.dimensionless_unscaled)
                    for field in fields["stars"]
                }
            elif cls == "gas":
                self.data["gas"] = {
                    field: (self.sim.gas[sim_field] * units["gas"][field] if sim_field in self.sim.gas.loadable_keys() else np.zeros(len(self.sim.gas)) * units["gas"].get(field, u.dimensionless_unscaled))
                    for field, sim_field in fields["gas"].items()
                }
            elif cls == "dm":
                self.data["dm"] = {
                    "mass": self.sim.dm.get(fields["dm"]["mass"], np.zeros(len(self.sim.dm))) * units["dm"]["mass"],
                }

        self.logger.info(f"NIHAO snapshot and halo data loaded successfully for classes: {load_classes}.")

    def get_halo_data(self):
        """Load halo data if available."""
        if self.halo_path:
            halos = self.sim.halos(filename=self.halo_path)
            self.halo_data = halos[1]
            self.logger.info("Halo data loaded.")
        else:
            self.halo_data = None
            self.logger.warning("No halo file provided or found.")
        return self.halo_data

    def get_galaxy_data(self):
        """Return basic galaxy data."""
        return {
            "redshift": self.nihao_config.get("galaxy", {}).get("redshift", 0.1),
            "center": self.center,
            "halfmassrad_stars": self.nihao_config["galaxy"]["halfmassrad_stars"],
        }

    def get_particle_data(self):
        """Return particle data."""
        return self.data

    def get_simulation_metadata(self):
        """Return metadata for the simulation."""
        return {"path": self.path, "halo_path": self.halo_path, "logger": str(self.logger)}

    def get_units(self):
        """Define and return units for all quantities based on the YAML config."""
        unit_map = {
            "Msun": u.M_sun,
            "Gyr": u.Gyr,
            "Zsun": u.Unit("Zsun"),
            "kpc": u.kpc,
            "km/s": u.km / u.s,
            "Msun/kpc^3": u.M_sun / u.kpc**3,
            "Msun/yr": u.M_sun / u.yr,
            "erg/g": u.erg / u.g,
            "K": u.K,
            "dimensionless": u.dimensionless_unscaled,
        }
        return {
            category: {field: unit_map[unit] for field, unit in self.nihao_config["units"][category].items()}
            for category in self.nihao_config["units"]
        }
