from .illustris import IllustrisHandler
from .nihao import NihaoHandler
from .base import BaseHandler

def get_input_handler(config: dict, logger=None) -> BaseHandler:
    """Creates a handler based on the config"""
    if logger is None:
        import logging
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

    simulation_name = config["simulation"]["name"]
    logger.info(f"Simulation name received: {simulation_name}")
    
    if simulation_name == "IllustrisTNG":
        logger.info("Using IllustrisHandler")
        return IllustrisHandler(**config["simulation"]["args"], logger=logger)
    elif simulation_name == "NIHAO":
        logger.info("Using NihaoHandler")        
        simulation_args = config["simulation"]["args"]
        if "galaxy" in config and "dist_z" in config["galaxy"]:
            simulation_args["dist_z"] = config["galaxy"]["dist_z"] 
        return NihaoHandler(**simulation_args, logger=logger)
    else:
        logger.error(f"Simulation {config['simulation']} is not supported")
        raise ValueError(f"Simulation {config['simulation']} is not supported")
