import pytest
import sys
import os
sys.path.insert(0, '/insert/your/project/path/here')
from rubix.galaxy.input_handler.nihao import NihaoHandler

import yaml

def load_config():
    """Load the configuration from the YAML file."""
    config_path = os.path.join(os.path.dirname(__file__), "../../../config/nihao_config.yml")
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

# Test configuration
TEST_CONFIG = {
    "data_path": "/insert/your/data/path/here",
    "halo_path": "/insert/your/halo/path/here",
    "units": {
        "stars": {"mass": "Msun", "age": "Gyr"},
        "gas": {"density": "Msun/kpc^3"},
        "dm": {"mass": "Msun"}
    }
}

def test_nihao_handler_initialization():
    """Tests the correct initialization of the NihaoHandler."""
    handler = NihaoHandler(path=TEST_CONFIG["data_path"], halo_path=TEST_CONFIG["halo_path"])
    assert handler.path == TEST_CONFIG["data_path"]
    assert handler.halo_path == TEST_CONFIG["halo_path"]

def test_load_data():
    """Tests data loading."""
    handler = NihaoHandler(path=TEST_CONFIG["data_path"], halo_path=TEST_CONFIG["halo_path"])
    handler.load_data()
    assert "stars" in handler.data
    assert "gas" in handler.data
    assert "dm" in handler.data

def test_get_galaxy_data():
    """Tests the return of galaxy data."""
    handler = NihaoHandler(path=TEST_CONFIG["data_path"])
    galaxy_data = handler.get_galaxy_data()
    assert isinstance(galaxy_data, dict)
    assert "redshift" in galaxy_data
    assert "center" in galaxy_data

def test_get_units():
    """Tests if the units are correctly defined."""
    handler = NihaoHandler(path=TEST_CONFIG["data_path"])
    units = handler.get_units()
    assert "stars" in units
    assert "mass" in units["stars"]
    assert units["stars"]["mass"].to_string() in ["Msun", "solMass"]