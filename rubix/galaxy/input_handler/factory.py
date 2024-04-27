from .base import BaseHandler
from .illustris import IllustrisHandler

__all__ = ["IllustrisHandler", "BaseHandler"]



def get_input_handler(config: dict) -> BaseHandler:
    """Creates a handler based on the config"""
    if config["simulation"]["name"] == "IllustrisTNG":
        return IllustrisHandler(**config["simulation"]["args"])
    else:
        raise ValueError(f"Simulation {config['simulation']} is not supported")
    
    
