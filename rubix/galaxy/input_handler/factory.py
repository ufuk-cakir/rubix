import os
from .base import BaseHandler
from .illustris import IllustrisHandler

__all__ = ["IllustrisHandler", "BaseHandler"]


def get_input_handler(config: dict, logger=None) -> BaseHandler:
    """Creates a handler based on the config"""
    if config["data/simulation/name"] == "IllustrisTNG":
        return IllustrisHandler(**config["simulation"]["args"], logger=logger)

    if config["data/simulation/name"] == "IllustrisAPI":
        path = os.path.join(
            config["data/output_path"],
            "illustris_api_data",
            f"galaxy-id-{config['data/simulation/args/galaxy_id']}.hdf5",
        )
        return IllustrisHandler(path, logger=logger)
    else:
        raise ValueError(f"Simulation {config['simulation']} is not supported")
