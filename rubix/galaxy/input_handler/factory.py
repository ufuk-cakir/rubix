from .illustris import IllustrisHandler
from .nihao import NihaoHandler
from .base import BaseHandler

def get_input_handler(config: dict, logger=None) -> BaseHandler:
    """Creates a handler based on the config"""
    simulation_name = config["simulation"]["name"]
    print(f"Debug: Simulation name received = {simulation_name}")
    
    if simulation_name == "IllustrisTNG":
        print("Using IllustrisHandler")  # debug
        return IllustrisHandler(**config["simulation"]["args"], logger=logger)
    elif simulation_name == "NIHAO":
        print("Using NihaoHandler")  # debug
        return NihaoHandler(**config["simulation"]["args"], logger=logger)
    else:
        raise ValueError(f"Simulation {config['simulation']} is not supported")
