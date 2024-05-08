import pytest
from unittest.mock import patch, MagicMock
from rubix.spectra.ssp.grid import SSPGrid
import numpy as np
import jax.numpy as jnp


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


def test_get_lookup():
    # Create a mock SSPGrid instance
    age = jnp.array([1e9, 2e9, 3e9])
    metallicity = jnp.array([0.01, 0.02, 0.03])
    wavelength = jnp.array([400, 500, 600])
    flux = jnp.array(
        [
            [[0.5, 0.6, 0.7], [0.8, 0.9, 1.0], [1.1, 1.2, 1.3]],
            [[1.4, 1.5, 1.6], [1.7, 1.8, 1.9], [2.0, 2.1, 2.2]],
            [[2.3, 2.4, 2.5], [2.6, 2.7, 2.8], [2.9, 3.0, 3.1]],
        ]
    )
    ssp_grid = SSPGrid(age, metallicity, wavelength, flux)

    # Get the lookup function
    lookup = ssp_grid.get_lookup()

    # Test interpolation at specific metallicity and age values
    metallicity_value = 0.02
    age_value = 2e9
    interpolated_flux = lookup(metallicity_value, age_value)

    # Assert that the interpolated flux is within the expected range
    assert (interpolated_flux >= jnp.min(flux)).all()
    assert (interpolated_flux <= jnp.max(flux)).all()

    # Test interpolation at metallicity and age values outside the grid range
    metallicity_value = 0.04
    age_value = 4e9
    interpolated_flux = lookup(metallicity_value, age_value)

    # Assert that the interpolated flux is NaN (outside the grid range)
    assert jnp.isnan(interpolated_flux).all()

    # Test interpolation at metallicity and age values on the grid boundary
    metallicity_value = 0.01
    age_value = 3e9
    interpolated_flux = lookup(metallicity_value, age_value)

    # Assert that the interpolated flux is within the expected range
    assert (interpolated_flux >= jnp.min(flux)).all()
    assert (interpolated_flux <= jnp.max(flux)).all()

    # Assert that interpolation matches on the grid points
    interpolated_flux = lookup(metallicity[0], age[0])
    assert np.allclose(interpolated_flux, flux[0][0])
