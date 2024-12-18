import numpy as np
from rubix.telescope.apertures import (
    SQUARE_APERTURE,
    CIRCULAR_APERTURE,
    HEXAGONAL_APERTURE,
)
from rubix.telescope.base import BaseTelescope
from rubix.telescope.utils import calculate_wave_edges, calculate_wave_seq
from rubix.utils import read_yaml
import os
import warnings
from typing import Optional, Union

PATH = os.path.dirname(os.path.abspath(__file__))
TELESCOPE_CONFIG_PATH = os.path.join(PATH, "telescopes.yaml")


class TelescopeFactory:
    def __init__(self, telescopes_config: Optional[Union[dict, str]] = None):
        if telescopes_config is None:
            warnings.warn(
                "No telescope config provided, using default stored in {}".format(
                    TELESCOPE_CONFIG_PATH
                )
            )
            self.telescopes_config = read_yaml(TELESCOPE_CONFIG_PATH)
        elif isinstance(telescopes_config, str):
            self.telescopes_config = read_yaml(telescopes_config)
        else:
            self.telescopes_config = telescopes_config

    def create_telescope(self, name):
        if name not in self.telescopes_config:
            raise ValueError(f"Telescope {name} not found in config")
        config = self.telescopes_config[name]

        # Get some parameters from the config
        sbin = np.floor(config["fov"] / config["spatial_res"]).astype(int)
        aperture_region = self._get_aperture(config["aperture_type"], sbin)
        wave_seq = calculate_wave_seq(config["wave_range"], config["wave_res"])
        wave_edges = calculate_wave_edges(wave_seq, config["wave_res"])

        telescope = BaseTelescope(
            fov=config["fov"],
            spatial_res=config["spatial_res"],
            wave_range=config["wave_range"],
            wave_res=config["wave_res"],
            lsf_fwhm=config["lsf_fwhm"],
            signal_to_noise=config["signal_to_noise"],
            sbin=sbin,
            aperture_region=aperture_region,
            pixel_type=config["pixel_type"],
            wave_seq=wave_seq,
            wave_edges=wave_edges,
        )
        telescope.__class__.__name__ = name

        return telescope

    def _get_aperture(self, type, size):
        if type == "square":
            return SQUARE_APERTURE(size)
        elif type == "circular":
            return CIRCULAR_APERTURE(size)
        elif type == "hexagonal":
            return HEXAGONAL_APERTURE(size)
        else:
            raise ValueError(f"Unknown aperture type: {type}")
