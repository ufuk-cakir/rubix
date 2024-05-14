import pytest
from unittest.mock import patch, MagicMock
from rubix.spectra.ssp.grid import SSPGrid
import numpy as np


def test_convert_units():
    data = [1, 2, 3]
    from_units = "Gyr"
    to_units = "Myr"
    expected_result = [1000, 2000, 3000]

    result = SSPGrid.convert_units(data, from_units, to_units)
    assert np.allclose(result, expected_result)


def test_from_hdf5():
    config = {
        "format": "hdf5",
        "file_name": "test.hdf5",
        "fields": {
            "age": {"name": "age", "in_log": False, "units": "Gyr"},
            "metallicity": {"name": "metallicity", "in_log": False, "units": ""},
            "wavelength": {"name": "wavelength", "in_log": False, "units": "Angstrom"},
            "flux": {"name": "flux", "in_log": False, "units": "Lsun/Angstrom"},
        },
        "name": "TestSSPGrid",
    }
    file_location = "/path/to/files"

    with patch("rubix.spectra.ssp.grid.h5py.File") as mock_file:
        mock_instance = MagicMock()
        mock_file.return_value = mock_instance
        mock_instance.__enter__.return_value = mock_instance
        mock_instance.__getitem__.side_effect = lambda key: {
            "age": [1, 2, 3],
            "metallicity": [0.1, 0.2, 0.3],
            "wavelength": [4000, 5000, 6000],
            "flux": [0.5, 1.0, 1.5],
        }[key]

        result = SSPGrid.from_hdf5(config, file_location)

        assert isinstance(result, SSPGrid)
        assert np.allclose(result.age, [1, 2, 3])

        assert np.allclose(result.metallicity, [0.1, 0.2, 0.3])
        assert np.allclose(result.wavelength, [4000, 5000, 6000])
        assert np.allclose(result.flux, [0.5, 1.0, 1.5])


def test_from_hdf5_wrong_format():
    config = {
        "format": "wrong",
        "file_name": "test.hdf5",
        "fields": {
            "age": {"name": "age", "in_log": False, "units": "Gyr"},
            "metallicity": {"name": "metallicity", "in_log": False, "units": ""},
            "wavelength": {"name": "wavelength", "in_log": False, "units": "Angstrom"},
            "flux": {"name": "flux", "in_log": False, "units": "Lsun/Angstrom"},
        },
        "name": "TestSSPGrid",
    }
    file_location = "/path/to/files"
    with pytest.raises(ValueError) as e:
        result = SSPGrid.from_hdf5(config, file_location)
        assert result is None
    assert str(e.value) == "Configured file format is not HDF5."
