from rubix.galaxy import get_input_handler
from typing import Union
from rubix.utils import read_yaml
from rubix.galaxy import IllustrisAPI





def convert_to_rubix(config: Union[dict, str]):
    # Create the input handler based on the config and create rubix galaxy data
    if isinstance(config, str):
        config = read_yaml(config)
    
    # If the simulationtype is IllustrisAPI, get data from IllustrisAPI
    
    if config["data"]["name"] == "IllustrisAPI":
        api = IllustrisAPI(**config["data"]["args"])
        api.load_galaxy(**config["data"]["load_galaxy_args"])
        
        # Load the saved data into the input handler
    
    input_handler = get_input_handler(config)
    input_handler.to_rubix(output_path=config["output_path"])
    




