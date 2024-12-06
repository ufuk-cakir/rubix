import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from rubix.galaxy.input_handler.nihao import NihaoHandler

@pytest.fixture
def mock_simulation():
    """Mocked simulation data for testing."""
    mock = MagicMock()

    mock.stars = MagicMock()
    mock.stars['pos'] = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
    ])
    mock.stars['mass'] = np.array([1.0, 2.0, 3.0])

    mock.gas = MagicMock()
    mock.dm = MagicMock()
    mock.gas['density'] = np.array([0.1, 0.2, 0.3])
    mock.dm['mass'] = np.array([10.0, 20.0, 30.0])

    return mock

@pytest.fixture
def mock_config():
    """Mocked configuration for NihaoHandler."""
    return {
        'fields': {
            'stars': {
                'age': 'age',
                'mass': 'mass',
                'metallicity': 'metallicity',
                'coords': 'pos',
                'velocity': 'vel'
            },
            'gas': {
                'density': 'density',
                'temperature': 'temperature'
            },
            'dm': {
                'mass': 'mass'
            }
        },
        'particles': {
            'stars': {
                'coords': 'kpc',
                'mass': 'Msun',
                'age': 'Gyr',
                'velocity': 'km/s',
                'metallicity': 'dimensionless'
            },
            'gas': {
                'density': 'Msun/kpc^3',
                'temperature': 'K'
            },
            'dm': {
                'mass': 'Msun'
            }
        },
        'units': {
            'stars': {
                'coords': 'kpc',
                'mass': 'Msun',
                'age': 'Gyr',
                'velocity': 'km/s',
                'metallicity': 'dimensionless'
            },
            'gas': {
                'density': 'Msun/kpc^3',
                'temperature': 'K'
            },
            'dm': {
                'mass': 'Msun'
            },
            'galaxy': {
                'redshift': 'dimensionless',
                'center': 'kpc',
                'halfmassrad_stars': 'kpc'
            }
        },
        'galaxy': {
            'redshift': 0.1,
            'halfmassrad_stars': 5.0
        }
    }

@pytest.fixture
def handler_with_mock_data(mock_simulation, mock_config):
    """Fixture to initialize the NihaoHandler with mocked data."""
    mock_simulation.stars.get.return_value = np.array([0.1, 0.2, 0.3])
    mock_simulation.stars["pos"] = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]) 
    mock_simulation.stars["mass"] = np.array([1.0, 2.0, 3.0]) 

    mock_simulation.gas.loadable_keys.return_value = ["density", "temperature"] 
    mock_simulation.gas["density"] = np.array([1.0, 2.0, 3.0]) 
    mock_simulation.gas["temperature"] = np.array([100.0, 200.0, 300.0]) 

    import pynbody
    pynbody.load = MagicMock(return_value=mock_simulation)

    handler = NihaoHandler(path="mock_path", halo_path="mock_halo_path", config=mock_config)
    handler.sim = mock_simulation
    return handler

def test_nihao_handler_initialization(handler_with_mock_data):
    """Test initialization of NihaoHandler."""
    assert handler_with_mock_data is not None

def test_load_data(handler_with_mock_data):
    """Test if data is loaded correctly."""
    handler_with_mock_data.load_data()
    assert handler_with_mock_data.center is not None

def test_get_galaxy_data(handler_with_mock_data):
    """Test retrieval of galaxy data."""
    galaxy_data = handler_with_mock_data.get_galaxy_data()
    assert galaxy_data is not None

def test_get_units(handler_with_mock_data):
    """Test if units are correctly returned."""
    units = handler_with_mock_data.get_units()
    assert 'stars' in units
    assert 'gas' in units
    assert 'dm' in units

def test_gas_data_load(handler_with_mock_data):
    """Test loading of gas data."""
    gas_data = handler_with_mock_data.get_particle_data()
    assert gas_data is not None
    assert len(gas_data) > 0

def test_stars_data_load(handler_with_mock_data):
    """Test loading of stars data."""
    star_data = handler_with_mock_data.get_particle_data()
    assert star_data is not None
    assert len(star_data) > 0

def test_dm_data_load(handler_with_mock_data):
    """Test loading of dark matter data."""
    dm_data = handler_with_mock_data.get_particle_data()
    assert dm_data is not None
    assert len(dm_data) > 0
