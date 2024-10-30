from .base import BaseHandler
from .illustris import IllustrisHandler

__all__ = ["IllustrisHandler", "BaseHandler"]


def get_input_handler(config: dict, logger=None) -> BaseHandler:
    """
    Creates a handler based on the config

    Args:
        config (dict): Configuration for the handler
        logger (Logger): Logger object

    Returns:
        BaseHandler based on the config
    """
    if config["simulation"]["name"] == "IllustrisTNG":
        return IllustrisHandler(**config["simulation"]["args"], logger=logger)
    else:
        raise ValueError(f"Simulation {config['simulation']} is not supported")
