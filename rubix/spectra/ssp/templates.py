""" This module contains the supported templates for the SSP grid. """

from .grid import SSPGrid
import os

TEMPLATES_PATH = os.path.join(os.path.dirname(__file__), "templates")




def load_hdf5_template(config: dict) -> SSPGrid:
    """
    Load a SSP grid from a HDF5 file.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary.
    
    Returns
    -------
    SSPGrid
        The SSP grid.
    """
    
    
