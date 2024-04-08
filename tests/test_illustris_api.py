import pytest
import os
from rubix.galaxy._input_handler._illustris_api import IllustrisAPI
from unittest.mock import patch, MagicMock
import numpy as np


@pytest.fixture
def api_key():
    return os.getenv("ILLUSTRIS_API_KEY")


@pytest.fixture
def api_instance(api_key, tmp_path):
    save_data_path = tmp_path / "test_data"
    save_data_path.mkdir()
    return IllustrisAPI(api_key=api_key, save_data_path=str(save_data_path))


@pytest.mark.parametrize(
    "input_id,particle_type,field_name,expected_exception",
    [
        (11, "stars", "Masses", None),
        ("invalid_id", "stars", "Masses", ValueError),
        (11, "invalid_type", "Masses", ValueError),
    ],
)
def test_get_particle_data(
    api_instance, input_id, particle_type, field_name, expected_exception
):
    if expected_exception:
        with pytest.raises(expected_exception):
            api_instance.get_particle_data(input_id, particle_type, field_name)
    else:
        data = api_instance.get_particle_data(input_id, particle_type, field_name)
        assert isinstance(data, dict)
        assert field_name in data["PartType4"]


def test__init__():
    api = IllustrisAPI(api_key="test_key")
    assert api.headers == {"api-key": "test_key"}
    assert api.particle_type == "stars"
    assert api.snapshot == 99
    assert api.simulation == "TNG50-1"
    assert api.baseURL == "http://www.tng-project.org/api/TNG50-1/snapshots/99"


def test_get_api_key(api_key):
    assert api_key is not None


def test_get_particle_data_valid_input(api_instance):
    data = api_instance.get_particle_data(11, "stars", "Masses")
    assert isinstance(data, dict)
    assert "Masses" in data["PartType4"]


def test_empty_response():
    with pytest.raises(ValueError):
        key = "wrong_key"
        api = IllustrisAPI(api_key=key)
        result = api.get_subhalo(11)
        assert result is None


def test__get(api_instance):  # Use requests_mock directly as a parameter
    result = api_instance.get_subhalo(11)
    assert result is not None


def test__get_no_api_key():
    with pytest.raises(ValueError):
        api_instance = IllustrisAPI(api_key=None)
        api_instance._get("http://www.tng-project.org/api/TNG50-1/snapshots/99")


def test_load_hdf5_valid_filename(api_instance):
    filename = "valid_filename"
    with patch("h5py.File", MagicMock()) as mock_file, patch(
        "os.path.exists", return_value=True
    ):
        mock_hdf5 = MagicMock()
        mock_group = MagicMock()
        mock_hdf5.keys.return_value = ["PartType4"]
        mock_hdf5.__getitem__.return_value = mock_group
        mock_group.keys.return_value = ["Masses"]
        mock_group.__getitem__.return_value = np.array([1, 2, 3])
        mock_file.return_value.__enter__.return_value = mock_hdf5

        data = api_instance._load_hdf5(filename)
        assert isinstance(data, dict)
        assert "PartType4" in data
        assert "Masses" in data["PartType4"]
        np.testing.assert_array_equal(data["PartType4"]["Masses"], np.array([1, 2, 3]))


def test_load_hdf5_filename_with_extension(api_instance):
    filename = "valid_filename.hdf5"
    with patch("h5py.File", MagicMock()) as mock_file, patch(
        "os.path.exists", return_value=True
    ):
        mock_hdf5 = MagicMock()
        mock_group = MagicMock()
        mock_hdf5.keys.return_value = ["PartType4"]
        mock_hdf5.__getitem__.return_value = mock_group
        mock_group.keys.return_value = ["Masses"]
        mock_group.__getitem__.return_value = np.array([1, 2, 3])
        mock_file.return_value.__enter__.return_value = mock_hdf5

        data = api_instance._load_hdf5(filename)
        assert isinstance(data, dict)
        assert "PartType4" in data
        assert "Masses" in data["PartType4"]
        np.testing.assert_array_equal(data["PartType4"]["Masses"], np.array([1, 2, 3]))


def test_load_hdf5_invalid_filename(api_instance):
    with pytest.raises(ValueError):
        api_instance._load_hdf5("invalid_filename")


def test_get_particle_data_empty_fields(api_instance):
    with pytest.raises(ValueError, match="Fields should not be empty."):
        api_instance.get_particle_data(0, "stars", "")


def test_load_galaxy_valid_input(api_instance):
    api_instance.DEFAULT_FIELDS = {
        "PartType0": [
            "ParticleIDs",
        ],
        "PartType4": [
            "ParticleIDs",
        ],
    }
    data = api_instance.load_galaxy(id=11, verbose=True)
    assert isinstance(data, dict)
    assert "SubhaloData" in data
    assert "PartType0" in data
    assert "PartType4" in data


def test_load_galaxy_invalid_id(api_instance):
    with pytest.raises(ValueError):
        api_instance.load_galaxy(id="invalid_id", verbose=True)
