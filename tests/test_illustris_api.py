import h5py
import pytest
import os
from rubix.galaxy._input_handler._illustris_api import IllustrisAPI

def test__init__():
    api = IllustrisAPI(api_key="test_key")
    assert api.headers == {"api-key": "test_key"}
    assert api.particle_type == "stars"
    assert api.snapshot == 99
    assert api.simulation == "TNG50-1"
    assert api.baseURL == "http://www.tng-project.org/api/TNG50-1/snapshots/99"


def test_get_api_key():
    key = os.getenv("ILLUSTRIS_API_KEY")
    assert key != None
    
def test_get_particle_data_valid_input():
    key = os.getenv("ILLUSTRIS_API_KEY")
    api = IllustrisAPI(api_key=key)
    data = api.get_particle_data(11, "stars", "Masses")
    assert isinstance(data, dict)
    assert "Masses" in data["PartType4"]


def test_get_particle_data_invalid_id():
    key = os.getenv("ILLUSTRIS_API_KEY")
    api = IllustrisAPI(api_key=key)
    with pytest.raises(ValueError):
        api.get_particle_data("invalid_id", "stars", "Masses")

def test_get_particle_data_invalid_particle_type():
    key = os.getenv("ILLUSTRIS_API_KEY")
    api = IllustrisAPI(api_key=key)
    with pytest.raises(ValueError):
        api.get_particle_data(11, "invalid_type", "Masses")

def test_empty_response():
    with pytest.raises(ValueError):
        key = "wrong_key"
        api = IllustrisAPI(api_key=key)
        result = api.get_subhalo(11)
        assert result == None

def test__get():  # Use requests_mock directly as a parameter
    key = os.getenv("ILLUSTRIS_API_KEY")
    api = IllustrisAPI(api_key=key)
    result = api.get_subhalo(11)
    assert result != None

def test__get_no_api_key():
    with pytest.raises(ValueError):
        api = IllustrisAPI(api_key=None)
        api._get("http://www.tng-project.org/api/TNG50-1/snapshots/99")
    
        
        
def test_load_hdf5_valid_filename():
    api = IllustrisAPI(api_key="test_key")
    filename = "valid_filename"
    # Create a dummy hdf5 file for testing
    with h5py.File(os.path.join(api.DATAPATH, f"{filename}.hdf5"), "w") as f:
        stars = f.create_group("PartType4")
        stars.create_dataset("Masses", data=[1, 2, 3])
        
    assert os.path.exists(os.path.join(api.DATAPATH, f"{filename}.hdf5"))
    data = api._load_hdf5(filename)
    assert isinstance(data, dict)
    assert "PartType4" in data

def test_load_hdf5_filename_with_extension():
    api = IllustrisAPI(api_key="test_key")
    filename = "valid_filename.hdf5"
    # Create a dummy hdf5 file for testing
    with h5py.File(os.path.join(api.DATAPATH, f"{filename}"), "w") as f:
        stars = f.create_group("PartType4")
        stars.create_dataset("Masses", data=[1, 2, 3])
    data = api._load_hdf5(filename)
    assert isinstance(data, dict)
    assert "PartType4" in data

def test_load_hdf5_invalid_filename():
    api = IllustrisAPI(api_key="test_key")
    with pytest.raises(ValueError):
        api._load_hdf5("invalid_filename") 

def test_get_particle_data_empty_fields():
    key = os.getenv("ILLUSTRIS_API_KEY")
    api = IllustrisAPI(api_key=key)
    with pytest.raises(ValueError, match="Fields should not be empty."):
        api.get_particle_data(0, "stars", "") 
def test_load_galaxy_valid_input():
    key = os.getenv("ILLUSTRIS_API_KEY")
    api = IllustrisAPI(api_key=key)
    api.DEFAULT_FIELDS = {
        "PartType0": [
            "ParticleIDs",
        ],
        "PartType4": [
            "ParticleIDs",
        ],
    }
    data = api.load_galaxy(id=11, verbose=True)
    assert isinstance(data, dict)
    assert "SubhaloData" in data
    assert "PartType0" in data 
    assert "PartType4" in data

def test_load_galaxy_invalid_id():
    key = os.getenv("ILLUSTRIS_API_KEY")
    api = IllustrisAPI(api_key=key)
    with pytest.raises(ValueError):
        api.load_galaxy(id="invalid_id", verbose=True)
