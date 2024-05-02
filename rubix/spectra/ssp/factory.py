from rubix.utils import read_yaml
from rubix.spectra.ssp.grid import SSPGrid
import os


def get_ssp_template(name: str)-> SSPGrid:
    """
    Get the SSP template from the configuration file.
    
    Returns
    -------
    SSPGrid
        The SSP template.
    """
    PATH = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(PATH, "ssp_config.yml")
    config = read_yaml(config_path) # TODO change this to load from rubix config
    
    config = config["templates"]
    # Check if the template exists in config
    if name not in config:
        raise ValueError(f"SSP template {name} not found in the supported configuration file.")
    
    
    if config[name]["format"].lower() == "hdf5":
        try:
            return SSPGrid.from_hdf5(config[name])
        except ValueError as e:
            raise ValueError(f"Error loading SSP template {name}: {e}")
    
    else:
        raise ValueError("Currently only HDF5 format is supported for SSP templates.")
        
    