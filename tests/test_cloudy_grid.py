import pytest
import unittest
import tempfile
import pickle
import jax.numpy as jnp
import numpy as np
from unittest.mock import patch
from rubix.spectra.cloudy.grid import CloudyGasLookup


class MockRubixData:
    class Gas:
        def __init__(self):
            self.metallicity = jnp.array([0.1, 0.2, 0.3])
            self.density = jnp.array([1.0, 2.0, 6.0])
            self.temperature = jnp.array([10000, 20000, 30000])
            self.mass = jnp.array([10.0, 40.0, 30.0])
            self.intr_emissivity = None
            self.luminosity = None
            self.internal_energy = jnp.array([1.0, 2.0, 3.0])
            self.electron_abundance = jnp.array([0.1, 0.4, 0.9])
            self.dispersionfactor = None
            self.spectra = None
            self.wavelengthrange = None

    class Galaxy:
        def __init__(self):
            self.redshift = 0.5

    def __init__(self):
        self.gas = self.Gas()
        self.galaxy = self.Galaxy()


class MockCloudyGasLookup(CloudyGasLookup):
    def __init__(self):
        # Mock data for testing
        self.line_names = [
            "MG 2 2802.71A",
            "O 3 5007.00A",
            "H 1 6563.00A",
            "N 2 6583.00A",
            "S 2 6716.00A",
        ]
        self.redshift_grid = jnp.array([0.0, 0.5, 1.0])
        self.metallicity_grid = jnp.array([-2.0, -1.0, 0.0])
        self.hden_grid = jnp.array([0.0, 1.0, 2.0])
        self.temp_grid = jnp.array([4.0, 4.5, 5.0])
        self.line_emissivity = [jnp.ones((3, 3, 3, 3)) for _ in range(5)]

    def find_nearest(self, grid, value):
        return jnp.argmin(jnp.abs(grid - value))

    def illustris_gas_temp(self, rubixdata):
        # Mock implementation
        return rubixdata

    def get_intrinsic_emissivity(self, rubixdata):
        # Mock implementation to set intrinsic emissivity
        rubixdata.gas.intr_emissivity = jnp.array(
            [
                [1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0],
            ]
        )
        return rubixdata


@pytest.fixture
def mock_rubixdata():
    return MockRubixData()


@pytest.fixture
def mock_cloudy_gas_lookup():
    return MockCloudyGasLookup()


def test_cloudy_gas_lookup_class_exists():
    assert "CloudyGasLookup" in globals(), "CloudyGasLookup class does not exist"


def test_load_anything():
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        # Data to be pickled
        data = {"key": "value"}

        # Pickle the data and write it to the temporary file
        with open(temp_file.name, "wb") as f:
            pickle.dump(data, f)

        # Use the load_anything method to load the data
        loaded_data = CloudyGasLookup.load_anything(temp_file.name)

        # Assert that the loaded data is the same as the original data
        assert loaded_data == data, "Loaded data does not match the original data"


@patch.object(CloudyGasLookup, "load_anything")
def test_init(mock_load_anything):
    # Define the mock data to be returned by load_anything
    mock_data = {
        "line_name": ["H_alpha", "OIII"],
        "line_emissivity": [1.0, 2.0],
        "redshift": [0.1, 0.2],
        "metals": [0.01, 0.02],
        "hden": [100, 200],
        "temp": [10000, 20000],
    }
    mock_load_anything.return_value = mock_data

    # Create an instance of CloudyGasLookup
    datafile_path = "dummy_path"
    instance = CloudyGasLookup(datafile_path)

    # Assert that the attributes are correctly initialized
    assert instance.datafile_path == datafile_path
    assert instance.data == mock_data
    assert instance.line_names == mock_data["line_name"]
    assert instance.line_emissivity == mock_data["line_emissivity"]
    assert instance.redshift_grid == mock_data["redshift"]
    assert instance.metallicity_grid == mock_data["metals"]
    assert instance.hden_grid == mock_data["hden"]
    assert instance.temp_grid == mock_data["temp"]


@patch.object(CloudyGasLookup, "load_anything")
def test_get_line_names(mock_load_anything):
    # Define the mock data to be returned by load_anything
    mock_data = {
        "line_name": ["H_alpha", "OIII"],
        "line_emissivity": [1.0, 2.0],
        "redshift": [0.1, 0.2],
        "metals": [0.01, 0.02],
        "hden": [100, 200],
        "temp": [10000, 20000],
    }
    mock_load_anything.return_value = mock_data

    # Create an instance of CloudyGasLookup
    datafile_path = "dummy_path"
    instance = CloudyGasLookup(datafile_path)

    # Call the get_line_names method
    line_names = instance.get_line_names()

    # Assert that the returned line names match the expected line names
    assert (
        line_names == mock_data["line_name"]
    ), "Returned line names do not match the expected line names"


def test_get_wavelength():
    # Create an instance of CloudyGasLookup
    datafile_path = "dummy_path"
    instance = CloudyGasLookup.__new__(CloudyGasLookup)  # Bypass __init__

    # Test the get_wavelength method
    assert (
        instance.get_wavelength("O  6 1031.91A") == 1031.91
    ), "Wavelength extraction failed for 'O  6 1031.91A'"
    assert (
        instance.get_wavelength("H_alpha 6562.8A") == 6562.8
    ), "Wavelength extraction failed for 'H_alpha 6562.8A'"


def test_get_all_wavelengths(mock_cloudy_gas_lookup):
    expected_wavelengths = jnp.array([2802.71, 5007.00, 6563.00, 6583.00, 6716.00])
    wavelengths = mock_cloudy_gas_lookup.get_all_wavelengths()
    assert jnp.allclose(wavelengths, expected_wavelengths)


def test_find_nearest():
    # Create an instance of CloudyGasLookup
    datafile_path = "dummy_path"
    instance = CloudyGasLookup.__new__(CloudyGasLookup)  # Bypass __init__

    # Define a test array and value
    test_array = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    test_value = 3.3

    # Call the find_nearest method
    nearest_index = instance.find_nearest(test_array, test_value)

    # Assert that the returned index is correct
    assert nearest_index == 2, f"Expected index 2, but got {nearest_index}"


def test_illustris_gas_temp():
    class MockGas:
        def __init__(self, internal_energy, electron_abundance):
            self.internal_energy = internal_energy
            self.electron_abundance = electron_abundance
            self.temperature = None

    class MockRubixData:
        def __init__(self, gas):
            self.gas = gas

    # Create mock data
    internal_energy = 1.0  # in km^2/s^2
    electron_abundance = 0.1
    mock_gas = MockGas(internal_energy, electron_abundance)
    mock_rubixdata = MockRubixData(mock_gas)

    # Create an instance of the class containing illustris_gas_temp
    instance = CloudyGasLookup.__new__(
        CloudyGasLookup
    )  # Replace with the actual class name

    # Call the illustris_gas_temp method
    result = instance.illustris_gas_temp(mock_rubixdata)

    # Check the temperature
    expected_temperature = (
        (5 / 3 - 1)
        * internal_energy
        / 1.38064852e-16
        * (4.0 / (1 + 3 * 0.76 + 4 * 0.76 * electron_abundance) * 1.6726219e-24)
    )
    assert (
        result.gas.temperature == expected_temperature
    ), f"Expected {expected_temperature}, but got {result.gas.temperature}"


def test_get_intrinsic_emissivity(mock_cloudy_gas_lookup, mock_rubixdata):
    result = mock_cloudy_gas_lookup.get_intrinsic_emissivity(mock_rubixdata)
    assert result.gas.intr_emissivity is not None
    assert result.gas.intr_emissivity.shape == (3, 5)
    assert jnp.all(result.gas.intr_emissivity == 1.0)


def test_get_luminosity(mock_cloudy_gas_lookup, mock_rubixdata):
    result = mock_cloudy_gas_lookup.get_luminosity(mock_rubixdata)
    assert result.gas.luminosity is not None
    assert result.gas.luminosity.shape == (3, 5)
    expected_luminosity = jnp.array(
        [
            [10.0, 10.0, 10.0, 10.0, 10.0],
            [20.0, 20.0, 20.0, 20.0, 20.0],
            [5.0, 5.0, 5.0, 5.0, 5.0],
        ]
    )
    assert jnp.allclose(result.gas.luminosity, expected_luminosity)


def test_dispersionfactor(mock_cloudy_gas_lookup, mock_rubixdata):
    result = mock_cloudy_gas_lookup.dispersionfactor(mock_rubixdata)
    assert result.gas.dispersionfactor is not None

    # Constants
    k_B = 1.3807e-16  # cm2 g s-2 K-1
    c = 2.99792458e10  # cm s-1
    m_p = 1.6726e-24  # g

    # Expected wavelengths
    wavelengths = jnp.array([2802.71, 5007.00, 6563.00, 6583.00, 6716.00])
    # wavelengths = jnp.array([1,1,1,1,1])
    # Calculate expected dispersion
    expected_dispersionfactor = jnp.sqrt(
        (8 * k_B * mock_rubixdata.gas.temperature * jnp.log(2)) / (m_p * c**2)
    )
    expected_dispersion = expected_dispersionfactor[:, None] * wavelengths

    assert jnp.allclose(result.gas.dispersionfactor, expected_dispersion)
    assert result.gas.dispersionfactor.shape == (3, 5)
    assert result.gas.dispersionfactor.shape == expected_dispersion.shape


def test_gaussian():
    # Create an instance of the class containing the gaussian method
    instance = CloudyGasLookup.__new__(CloudyGasLookup)
    x = jnp.array([0.0, 1.0, 2.0])
    a = 1.0
    b = 1.0
    c = 1.0
    expected_output = jnp.array([0.60653066, 1.0, 0.60653066])
    output = instance.gaussian(x, a, b, c)
    assert jnp.allclose(
        output, expected_output
    ), f"Expected {expected_output}, but got {output}"


def test_get_wavelengthrange():
    # Create an instance of the class containing the get_wavelengthrange method
    instance = CloudyGasLookup.__new__(CloudyGasLookup)
    instance.line_names = ["O  6 400.00A", "MG 2 500.00A", "MG 2 700.00A"]
    instance.get_wavelengths = lambda: jnp.array([400, 700])

    steps = 4000
    expected_start = 400 * 0.9
    expected_end = 700 * 1.1
    expected_wavelengthrange = jnp.linspace(expected_start, expected_end, steps)

    output = instance.get_wavelengthrange(steps)

    assert output.shape == (
        steps,
    ), f"Expected shape {(steps,)}, but got {output.shape}"
    assert jnp.allclose(
        output, expected_wavelengthrange
    ), f"Expected {expected_wavelengthrange}, but got {output}"


def test_get_spectra(mock_cloudy_gas_lookup, mock_rubixdata):
    # Call the get_spectra function
    result = mock_cloudy_gas_lookup.get_spectra(mock_rubixdata)

    # Check that the spectra and wavelengthrange are not None
    assert result.gas.spectra is not None
    assert result.gas.wavelengthrange is not None

    # Check the shape of the spectra
    assert len(result.gas.spectra) == len(mock_rubixdata.gas.luminosity)
    assert result.gas.spectra[0].shape == (4000,)

    # Check that the sum of the spectrum is larger than 0
    for spectrum in result.gas.spectra:
        assert jnp.sum(spectrum) > 0, "The sum of the spectrum should be larger than 0"
