from rubix.galaxy import get_input_handler
from typing import Union
from rubix.utils import read_yaml
from rubix.galaxy import IllustrisAPI
from rubix.utils import load_galaxy_data
from rubix.logger import get_logger
from rubix.galaxy.alignment import center_particles
import os





def convert_to_rubix(config: Union[dict, str]):
    # Create the input handler based on the config and create rubix galaxy data
    if isinstance(config, str):
        config = read_yaml(config)
        
    # Setup a logger based on the config
    logger_config = config["logger"] if "logger" in config else None
    
    logger = get_logger(logger_config)
    
    # If the simulationtype is IllustrisAPI, get data from IllustrisAPI
    
    if config["data"]["name"] == "IllustrisAPI":
        logger.info("Loading data from IllustrisAPI")
        api = IllustrisAPI(**config["data"]["args"], logger=logger)
        api.load_galaxy(**config["data"]["load_galaxy_args"])
        
        # Load the saved data into the input handler
    logger.info("Loading data into input handler")
    input_handler = get_input_handler(config,logger=logger)
    input_handler.to_rubix(output_path=config["output_path"])
    
    return config["output_path"]
    

def prepare_input(config: Union[dict, str]):
    
    file_path = config["output_path"]
    file_path = os.path.join(file_path, "rubix_galaxy.h5")
    
    # Load the data from the file
    data = load_galaxy_data(file_path)
    
    
    stellar_coordinates = data["particle_data"]["stars"]["coords"]
    stellar_velocities = data["particle_data"]["stars"]["velocities"]
    galaxy_center = data["particle_data"]["subhalo_center"]
    
    # Center the particles
    new_stellar_coordinates, new_stellar_velocities = center_particles(stellar_coordinates, stellar_velocities, galaxy_center)
    
    # Load the metallicity and age data
    
    stars_metallicity = data["particle_data"]["stars"]["metallicity"]
    stars_mass = data["particle_data"]["stars"]["mass"]
    stars_age = data["particle_data"]["stars"]["age"]
    
    return new_stellar_coordinates, new_stellar_velocities, stars_metallicity, stars_mass, stars_age
    
    
    