import pytest
import requests
from unittest.mock import patch, MagicMock
from rubix.spectra.ssp.grid import SSPGrid, HDF5SSPGrid, pyPipe3DSSPGrid
import numpy as np
import jax.numpy as jnp
import os
from astropy.io import fits


def test_convert_units():
    data = [1, 2, 3]
    from_units = "Gyr"
    to_units = "Myr"
    expected_result = [1000, 2000, 3000]

    result = SSPGrid.convert_units(data, from_units, to_units)
    assert np.allclose(result, expected_result)

def test_SSPGrid_from_file():
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

    result = SSPGrid.from_file(config, file_location)

    assert isinstance(result, SSPGrid)

def test_from_hdf5():
    config = {
        "format": "hdf5",
        "file_name": "test.hdf5",
        "source": "http://example.com/template.hdf5",
        "fields": {
            "age": {"name": "age", "in_log": False, "units": "Gyr"},
            "metallicity": {"name": "metallicity", "in_log": False, "units": ""},
            "wavelength": {"name": "wavelength", "in_log": False, "units": "Angstrom"},
            "flux": {"name": "flux", "in_log": False, "units": "Lsun/Angstrom"},
        },
        "name": "TestSSPGrid",
    }
    file_location = "/path/to/files"

    with patch("os.path.exists") as mock_exists, \
         patch("rubix.spectra.ssp.grid.h5py.File") as mock_file:
        
        mock_exists.return_value = True
        mock_instance = MagicMock()
        mock_file.return_value = mock_instance
        mock_instance.__enter__.return_value = mock_instance
        mock_instance.__getitem__.side_effect = lambda key: {
            "age": [1, 2, 3],
            "metallicity": [0.1, 0.2, 0.3],
            "wavelength": [4000, 5000, 6000],
            "flux": [0.5, 1.0, 1.5],
        }[key]

        result = HDF5SSPGrid.from_file(config, file_location)

        assert isinstance(result, HDF5SSPGrid)
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
        result = HDF5SSPGrid.from_file(config, file_location)
        assert result is None
    assert str(e.value) == "Configured file format is not HDF5."

@pytest.fixture
def ssp_grid():
    # Create a sample SSP grid
    age = [1, 2, 3]
    metallicity = [0.1, 0.2, 0.3]
    wavelength = [4000, 5000, 6000]
    flux = [
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
        [[19, 20, 21], [22, 23, 24], [25, 26, 27]],
    ]
    return pyPipe3DSSPGrid(age, metallicity, wavelength, flux)

def test_get_wavelength_from_header(ssp_grid):
    header = fits.Header()
    header["CRVAL1"] = 4000
    header["CDELT1"] = 1000
    header["NAXIS1"] = 3
    header["CRPIX1"] = 1
    wavelength = ssp_grid.get_wavelength_from_header(header)
    assert np.allclose(wavelength, [4000, 5000, 6000])

def test_get_wavelength_from_header_no_cdelt(ssp_grid):
    header = fits.Header()
    header["CRVAL1"] = 4000
    header["CDELT1"] = None
    header["NAXIS1"] = 3
    header["CRPIX1"] = 1
    wavelength = ssp_grid.get_wavelength_from_header(header)
    assert np.allclose(wavelength, [4000, 4001, 4002])

#def test_get_normalization_wavelength(ssp_grid):
#    header = fits.Header()
#    header["WAVENORM"] = 5000
#    wavelength = [4000, 5000, 6000]
#    flux_models = [
#        [0.9, 1.1, 1.2],
#        [1.0, 1.2, 1.1],
#        [1.1, 1.3, 1.4],
#    ]
#    n_models = 3
#    normalization_wavelength = ssp_grid.get_normalization_wavelength(header, wavelength, flux_models, n_models)
#    assert normalization_wavelength == 5000

def test_get_tZ_models(ssp_grid):
    header = fits.Header()
    header["NAME0"] = "spec_ssp_1.0_z01.spec"
    header["NAME1"] = "spec_ssp_2.0_z02.spec"
    header["NAME2"] = "spec_ssp_3.0_z03.spec"
    header["NORM0"] = 1.0
    header["NORM1"] = 1.0
    header["NORM2"] = 1.0
    n_models = 3
    ages, metallicities, m2l = ssp_grid.get_tZ_models(header, n_models)
    assert np.allclose(ages, [1.0, 2.0, 3.0])
    assert np.allclose(metallicities, [0.01, 0.02, 0.03])
    assert np.allclose(m2l, [1.0, 1.0, 1.0])

def test_get_tZ_models_zero_norm(ssp_grid):
    header = fits.Header()
    header["NAME0"] = "spec_ssp_1.0_z01.spec"
    header["NAME1"] = "spec_ssp_2.0_z02.spec"
    header["NORM0"] = 0.0
    header["NORM1"] = 1.0
    n_models = 2
    ages, metallicities, m2l = ssp_grid.get_tZ_models(header, n_models)
    assert np.allclose(ages, [1.0, 2.0])
    assert np.allclose(metallicities, [0.01, 0.02])
    assert np.allclose(m2l, [1.0, 1.0])

def test_get_tZ_models_yr_in_name(ssp_grid):
    header = fits.Header()
    header["NAME0"] = "spec_ssp_1.0Gyr_z01.spec"
    header["NORM0"] = 1.0
    n_models = 1
    ages, metallicities, m2l = ssp_grid.get_tZ_models(header, n_models)
    assert np.allclose(ages, [1.0])
    assert np.allclose(metallicities, [0.01])
    assert np.allclose(m2l, [1.0])

def test_from_pyPipe3D():
    config = {
        "format": "pypipe3d",
        "file_name": "test.fits",
        "source": "http://example.com/",
        "fields": {
            "age": {"units": "Gyr", "in_log": False},
            "metallicity": {"units": "", "in_log": False},
            "wavelength": {"units": "Angstrom", "in_log": False},
            "flux": {"units": "Lsun/Angstrom", "in_log": False},
        },
        "name": "pyPipe3DSSPGrid",
    }
    file_location = "/path/to/files"

    with patch("os.path.exists") as mock_exists, \
         patch("rubix.spectra.ssp.grid.fits.open") as mock_file:
        mock_exists.return_value = True

        mock_instance = MagicMock()
        mock_file.return_value = mock_instance
        mock_instance.__enter__.return_value = mock_instance
        mock_instance[0].header = {
            "CRVAL1": 4000,
            "CDELT1": 1000,
            "NAXIS1": 4,
            "CRPIX1": 1,
            "WAVENORM": 5000,
            "NAME0": "spec_ssp_1.0_z01.spec",
            "NAME1": "spec_ssp_2.0_z01.spec",
            "NAME2": "spec_ssp_3.0_z01.spec",
            "NAME3": "spec_ssp_1.0_z02.spec",
            "NAME4": "spec_ssp_2.0_z02.spec",
            "NAME5": "spec_ssp_3.0_z02.spec",
            "NORM0": 1.0,
            "NORM1": 1.0,
            "NORM2": 1.0,
            "NORM3": 1.0,
            "NORM4": 1.0,
            "NORM5": 1.0,
            "NAXIS2": 6
        }
        mock_instance[0].data = [[0.5, 1.0, 1.5, 2.0],[0.5, 1.0, 1.5, 2.0],[0.5, 1.0, 1.5, 2.0],[0.5, 1.0, 1.5, 2.0],[0.5, 1.0, 1.5, 2.0],[0.5, 1.0, 1.5, 2.0]]

        result = pyPipe3DSSPGrid.from_file(config, file_location)

        assert isinstance(result, pyPipe3DSSPGrid)
        assert np.allclose(result.age, [1, 2, 3])

        assert np.allclose(result.metallicity, [0.01, 0.02])
        assert np.allclose(result.wavelength, [4000, 5000, 6000, 7000])
        assert np.allclose(result.flux, [[[0.5, 1.0, 1.5, 2.0],[0.5, 1.0, 1.5, 2.0],[0.5, 1.0, 1.5, 2.0]],[[0.5, 1.0, 1.5, 2.0],[0.5, 1.0, 1.5, 2.0],[0.5, 1.0, 1.5, 2.0]]])
        assert result.flux.shape == (2, 3, 4)

def test_from_pyPipe3D_wrong_field_name():
    config = {
        "format": "wrong",
        "file_name": "test.fits",
        "format": "pypipe3d",
        "source": "http://example.com/",
        "fields": {
            "age": {"name": "age", "in_log": False, "units": "Gyr"},
            "metallicity": {"name": "metallicity", "in_log": False, "units": ""},
            "wavelength": {"name": "wavelength", "in_log": False, "units": "Angstrom"},
            "flux": {"name": "flux", "in_log": False, "units": "Lsun/Angstrom"},
            "wrong_field": {"wrong_field_name": "wrong_field_name", "in_log": False, "units": "wrong_units"},
        },
        "name": "TestSSPGrid",
    }
    file_location = "/path/to/files"

    with pytest.raises(ValueError) as e, \
         patch("os.path.exists") as mock_exists, \
         patch("rubix.spectra.ssp.grid.fits.open") as mock_file:
        mock_exists.return_value = True

        mock_instance = MagicMock()
        mock_file.return_value = mock_instance
        mock_instance.__enter__.return_value = mock_instance
        mock_instance[0].header = {
            "CRVAL1": 4000,
            "CDELT1": 1000,
            "NAXIS1": 3,
            "CRPIX1": 1,
            "WAVENORM": 5000,
            "NAME0": "spec_ssp_1.0_z01.spec",
            "NORM0": 1.0,
            "NAXIS2": 1
            }
        mock_instance[0].data = [[0.5, 1.0, 1.5]]

        result = pyPipe3DSSPGrid.from_file(config, file_location)
        assert result is None
        assert str(e.value) == f"Field wrong_field_name not recognized"


def test_from_pyPipe3D_wrong_format():
    config = {
        "format": "wrong",
        "file_name": "test.fits",
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
        result = pyPipe3DSSPGrid.from_file(config, file_location)
        assert result is None
    assert str(e.value) == "Configured file format is not fits."


def test_checkout_SSP_template():
    config = {
        "file_name": "ssp_template.fits",
        "source": "http://example.com/",
    }
    file_location = "/path/to/save"

    with patch("os.path.exists") as mock_exists, \
         patch("requests.get") as mock_get, \
         patch("builtins.open", create=True) as mock_open:
        
        mock_exists.return_value = False
        mock_get.return_value.status_code = 200
        mock_get.return_value.content = b"mock file content"

        file_path = SSPGrid.checkout_SSP_template(config, file_location)

        mock_exists.assert_called_once_with(os.path.join(file_location, config["file_name"]))
        mock_get.assert_called_once_with(config["source"] + config["file_name"])
        mock_open.assert_called_once_with(file_path, "wb")
        mock_open.return_value.__enter__.return_value.write.assert_called_once_with(b"mock file content")

        assert file_path == os.path.join(file_location, config["file_name"])

def test_checkout_SSP_template_file_exists():
    config = {
        "format": "hdf5",
        "file_name": "test.hdf5",
        "source": "http://example.com",
        "fields": {
            "age": {"name": "age", "in_log": False, "units": "Gyr"},
            "metallicity": {"name": "metallicity", "in_log": False, "units": ""},
            "wavelength": {"name": "wavelength", "in_log": False, "units": "Angstrom"},
            "flux": {"name": "flux", "in_log": False, "units": "Lsun/Angstrom"},
        },
        "name": "TestSSPGrid",
    }
    file_location = "/path/to/files"

    # Create a dummy file at the expected file path
    file_path = os.path.join(file_location, config["file_name"])

    with patch("os.path.exists") as mock_exists:
        mock_exists.return_value = True

        # Call the function
        result = HDF5SSPGrid.checkout_SSP_template(config, file_location)

        # Verify that the function returns the expected file path
        assert result == file_path


def test_checkout_SSP_template():
    config = {
        "file_name": "ssp_template.fits",
        "source": "http://example.com/",
    }
    file_location = "/path/to/save"

    with patch("os.path.exists") as mock_exists, \
         patch("requests.get") as mock_get, \
         patch("builtins.open", create=True) as mock_open:
        
        mock_exists.return_value = False
        mock_get.return_value.status_code = 200
        mock_get.return_value.content = b"Mock file content"

        file_path = SSPGrid.checkout_SSP_template(config, file_location)

        mock_exists.assert_called_once_with(os.path.join(file_location, config["file_name"]))
        mock_get.assert_called_once_with(config["source"] + config["file_name"])
        mock_open.assert_called_once_with(file_path, "wb")
        mock_open.return_value.__enter__.return_value.write.assert_called_once_with(b"Mock file content")

        assert file_path == os.path.join(file_location, config["file_name"])


def test_checkout_SSP_template_file_download_error():
    config = {
        "format": "hdf5",
        "file_name": "test.hdf5",
        "source": "http://example.com/", # This URL will raise an exception when accessed
        "fields": {
            "age": {"name": "age", "in_log": False, "units": "Gyr"},
            "metallicity": {"name": "metallicity", "in_log": False, "units": ""},
            "wavelength": {"name": "wavelength", "in_log": False, "units": "Angstrom"},
            "flux": {"name": "flux", "in_log": False, "units": "Lsun/Angstrom"},
        },
        "name": "TestSSPGrid",
    }
    file_location = "/path/to/files"

    # Mock the requests.get function to raise an exception
    with patch("requests.get") as mock_get:
        mock_get.side_effect = requests.exceptions.HTTPError("Download error")

        # Call the function and verify that it raises a ValueError
        try:
            HDF5SSPGrid.checkout_SSP_template(config, file_location)
            assert False, "Expected ValueError to be raised"
        except FileNotFoundError as e:
            assert str(e) == "Could not download file test.hdf5 from url http://example.com/."

def test_checkout_SSP_template_file_download_failed():
    config = {
        "format": "hdf5",
        "file_name": "test.hdf5",
        "source": "http://example.com/",
        "fields": {
            "age": {"name": "age", "in_log": False, "units": "Gyr"},
            "metallicity": {"name": "metallicity", "in_log": False, "units": ""},
            "wavelength": {"name": "wavelength", "in_log": False, "units": "Angstrom"},
            "flux": {"name": "flux", "in_log": False, "units": "Lsun/Angstrom"},
        },
        "name": "TestSSPGrid",
    }
    file_location = "/path/to/files"

    # Mock the requests.get function to return a failed response
    with patch("requests.get") as mock_get:
        mock_get.return_value.status_code = 404

        # Call the function and verify that it raises a FileNotFoundError
        try:
            HDF5SSPGrid.checkout_SSP_template(config, file_location)
            assert False, "Expected FileNotFoundError to be raised"
        except FileNotFoundError as e:
            assert str(e) == f"Could not download file {config['file_name']} from url {config['source']}."


def test_get_lookup_interpolation():
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
    lookup = ssp_grid.get_lookup_interpolation()

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

    # Assert that the interpolated flux is 0 (outside the grid range)
    assert jnp.all(interpolated_flux == 0)

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
