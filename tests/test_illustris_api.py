import pytest
import os
from virtual_telescope.galaxy._input_handler._illustris_api import IllustrisAPI

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


def test_empty_response():
    with pytest.raises(ValueError):
        key = "wrong_key"
        api = IllustrisAPI(api_key=key)
        result = api.get_subhalo(0)
        assert result == None

def test__get():  # Use requests_mock directly as a parameter
    key = os.getenv("ILLUSTRIS_API_KEY")
    api = IllustrisAPI(api_key=key)
    result = api.get_subhalo(0)
    assert result != None

def test__get_no_api_key():
    with pytest.raises(ValueError):
        api = IllustrisAPI(api_key=None)
        api._get("http://www.tng-project.org/api/TNG50-1/snapshots/99")
        
        