import pytest
from unittest.mock import patch, MagicMock
from rubix.galaxy.input_handler.nihao import NihaoHandler
import numpy as np

@pytest.fixture
def mock_simulation():
    """Create a mocked simulation object"""
    mock_sim = MagicMock()
    mock_sim.stars = {
        'pos': [[0, 0, 0], [1, 1, 1]],
        'mass': [1, 2],
        'vel': [[0, 0, 0], [1, 1, 1]],
        'metals': [0.1, 0.2],
        'tform': np.array([0.5, 0.6]) 
    }
    mock_sim.gas = {
        'rho': [1e-4, 1e-5],
        'temp': [1e3, 1e4],
        'metals': [0.1, 0.2]
    }
    mock_sim.dm = {
        'mass': [10, 20]
    }
    mock_sim.physical_units = MagicMock()
    return mock_sim


@pytest.fixture
def mock_config():
    """Create a mocked config object"""
    return {
        "data_path": "mock_path",
        "halo_path": "mock_halo_path",
        "fields": {
            "stars": {"age": "tform", "mass": "mass", "metallicity": "metals", "coords": "pos", "velocity": "vel"},
            "gas": {"density": "rho", "temperature": "temp", "metallicity": "metals"},
            "dm": {"mass": "mass"},
        },
        "units": {
            "stars": {"mass": "Msun", "age": "Gyr", "metallicity": "Zsun", "coords": "kpc", "velocity": "km/s"},
            "gas": {"density": "Msun/kpc^3", "temperature": "K", "metallicity": "Zsun"},
            "dm": {"mass": "Msun"},
            "galaxy": {"redshift": "dimensionless", "center": "kpc", "halfmassrad_stars": "kpc"},
        },
        "galaxy": {"redshift": 0.1, "center": [0, 0, 0], "halfmassrad_stars": 5.0},
        "load_classes": ["dm", "gas", "stars"],
    }


@pytest.fixture
def handler_with_mock_data(mock_simulation, mock_config):
    """Fixture to initialize the NihaoHandler with mocked data."""
    with patch('pynbody.load', return_value=mock_simulation):
        handler = NihaoHandler(path="mock_path", halo_path="mock_halo_path", config=mock_config) #add config=mock_config
        return handler


def test_nihao_handler_initialization(handler_with_mock_data):
    """Tests the correct initialization of the NihaoHandler"""
    handler = handler_with_mock_data
    assert handler.path == "mock_path"
    assert handler.halo_path == "mock_halo_path"

def test_load_data(handler_with_mock_data):
    """Tests if data for stars, gas, and dark matter are loaded correctly"""
    handler = handler_with_mock_data
    assert "stars" in handler.data
    assert "gas" in handler.data
    assert "dm" in handler.data

def test_get_galaxy_data(handler_with_mock_data):
    """Tests if galaxy data (e.g., redshift, center) are returned correctly"""
    handler = handler_with_mock_data
    galaxy_data = handler.get_galaxy_data()
    assert isinstance(galaxy_data, dict)
    assert "redshift" in galaxy_data
    assert "center" in galaxy_data

def test_get_units(handler_with_mock_data):
    """Tests if units are returned correctly."""
    handler = handler_with_mock_data
    units = handler.get_units()
    assert "stars" in units
    assert units["stars"]["mass"].to_string() in ["Msun", "solMass"]
    assert units["gas"]["density"].to_string() in ["Msun / kpc3", "solMass / kpc3"]
