import pytest
from unittest.mock import patch, MagicMock, Mock

# Assume the module name is rubix_conversion.py where the functions are located
from rubix.core.data import convert_to_rubix, prepare_input

# Mock configuration for tests
config_dict = {
    "data": {
        "name": "IllustrisAPI",
        "args": {"api_key": "your_api_key"},
        "load_galaxy_args": {"snapshot": "latest"},
    },
    "output_path": "/path/to/output",
    "logger": {"level": "DEBUG"},
}

config_path = "path/to/config.yaml"


# Test convert_to_rubix function
@patch("rubix.core.data.read_yaml")
@patch("rubix.core.data.get_logger")
@patch("rubix.core.data.get_input_handler")
@patch("rubix.core.data.IllustrisAPI")
def test_convert_to_rubix(
    mock_illustris_api, mock_input_handler, mock_logger, mock_read_yaml
):
    mock_read_yaml.return_value = config_dict
    mock_logger.return_value = MagicMock()
    mock_input_handler.return_value = MagicMock()
    mock_input_handler.return_value.to_rubix.return_value = None
    mock_illustris_api.return_value = MagicMock()
    mock_illustris_api.return_value.load_galaxy.return_value = None

    output_path = convert_to_rubix(config_path)
    assert output_path == config_dict["output_path"]
    mock_read_yaml.assert_called_once_with(config_path)
    mock_logger.assert_called_once()
    mock_input_handler.assert_called_once()
    mock_illustris_api.assert_called_once()


def test_rubix_file_already_exists():
    # Mock configuration for the test
    config = {
        "output_path": "/fake/path",
        "data": {"name": "IllustrisAPI", "args": {}, "load_galaxy_args": {}},
    }

    # Create a mock logger that does nothing
    mock_logger = Mock()

    with patch("rubix.core.data.os.path.exists", return_value=True) as mock_exists:
        with patch(
            "rubix.core.data.get_logger", return_value=mock_logger
        ) as mock_get_logger:
            # Call the function under test
            result = convert_to_rubix(config)

            # Check that the file existence check was performed correctly
            mock_exists.assert_called_once_with("/fake/path/rubix_galaxy.h5")

            # Check that the logger was created
            mock_get_logger.assert_called_once_with(None)

            # Ensure the function logs the right message and skips conversion
            mock_logger.info.assert_called_with(
                "Rubix galaxy file already exists, skipping conversion"
            )

            # Verify that the function returns the expected path without performing further actions
            assert (
                result == "/fake/path"
            ), "Function should return the output path when file exists"


# Test prepare_input function
@patch("rubix.core.data.os.path.join")
@patch("rubix.core.data.center_particles")
def test_prepare_input(mock_center_particles, mock_path_join):
    mock_path_join.return_value = "/path/to/output/rubix_galaxy.h5"
    particle_data = {
        "particle_data": {
            "stars": {
                "coords": [[1, 2, 3]],
                "velocity": [[4, 5, 6]],
                "metallicity": [0.1],
                "mass": [1000],
                "age": [4.5],
            },
            "gas": {
                "coords": [[7, 8, 9]],
                "velocity": [[10, 11, 12]],
                "metallicity": [0.2],
                "mass": [2000],
                "density": [1],
                "hsml": [0.1],
                "sfr": [0.1],
                "internal_energy": [0.1],
                "electron_abundance": [0.1],
                "metals": [1]
            }
        },
        "subhalo_center": [0, 0, 0],
    }
    units = {
        "galaxy": {"center": "kpc", "halfmassrad_stars": "kpc", "redshift": ""},
        "stars": {"mass": "Msun"},
        "gas": {"mass": "Msun", "density": "Msun/kpc^3", "hsml": "kpc", "sfr": "Msun/yr", "internal_energy": "erg/g", "electron_abundance": "dimensionless", "metals": "dimensionless"}
    }
    mock_load_galaxy_data = (particle_data, units)
    with patch("rubix.core.data.load_galaxy_data", return_value=mock_load_galaxy_data):
        #mock_center_particles_stars.return_value = ([[1, 2, 3]], [[4, 5, 6]])
        #mock_center_particles_gas.return_value = ([[7, 8, 9]], [[10, 11, 12]])
        mock_center_particles.return_value = ([[1, 2, 3], [7, 8, 9]], [[4, 5, 6], [10, 11, 12]])

        result = prepare_input(config_dict)

        all_coords, all_velocities, stars_metallicity, stars_mass, stars_age, gas_coords, gas_velocities, gas_metallicity, gas_mass, gas_density, gas_hsml, gas_sfr, gas_internal_energy, gas_electron_abundance, gas_metals = result

        stars_coords = all_coords[:len(particle_data["particle_data"]["stars"]["coords"])]
        gas_coords = all_coords[len(particle_data["particle_data"]["stars"]["coords"]):]

        stars_velocities = all_velocities[:len(particle_data["particle_data"]["stars"]["velocity"])]
        gas_velocities = all_velocities[len(particle_data["particle_data"]["stars"]["velocity"]):]

        assert stars_coords == [[1, 2, 3]]
        assert stars_velocities == [[4, 5, 6]]
        assert stars_metallicity == [0.1]
        assert stars_mass == [1000]
        assert stars_age == [4.5]

        assert gas_coords == [[7, 8, 9]]
        assert gas_velocities == [[10, 11, 12]]
        assert gas_metallicity == [0.2]
        assert gas_mass == [2000]
        assert gas_density == [1]
        assert gas_hsml == [0.1]
        assert gas_sfr == [0.1]
        assert gas_internal_energy == [0.1]
        assert gas_electron_abundance == [0.1]
        assert gas_metals == [1]

        assert mock_center_particles.call_count == 2

        #mock_path_join.assert_called_once_with(
        #    config_dict["output_path"], "rubix_galaxy.h5"
        #)
        #mock_center_particles.assert_called_once()

