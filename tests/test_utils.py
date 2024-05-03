import pytest  # type: ignore # noqa
import yaml
from rubix.utils import convert_values_to_physical, SFTtoAge, read_yaml
from astropy.cosmology import Planck15 as cosmo


def test_convert_values_to_physical():
    # Test with some arbitrary values
    value = 10
    a = 0.5
    a_scale_exponent = 2
    hubble_param = 0.7
    hubble_scale_exponent = 2
    CGS_conversion_factor = 1.0

    result = convert_values_to_physical(
        value,
        a,
        a_scale_exponent,
        hubble_param,
        hubble_scale_exponent,
        CGS_conversion_factor,
    )

    # Check if the result is as expected
    expected_result = (
        value
        * a**a_scale_exponent
        * hubble_param**hubble_scale_exponent
        * CGS_conversion_factor
    )
    assert result == expected_result, f"Expected {expected_result}, but got {result}"


def test_convert_values_to_physical_with_cgs_factor_zero():
    # Test with some arbitrary values
    value = 10
    a = 0.5
    a_scale_exponent = 2
    hubble_param = 0.7
    hubble_scale_exponent = 2
    CGS_conversion_factor = 0.0

    result = convert_values_to_physical(
        value,
        a,
        a_scale_exponent,
        hubble_param,
        hubble_scale_exponent,
        CGS_conversion_factor,
    )

    # Check if the result is as expected
    # When CGS_conversion_factor is 0, it should be treated as 1
    expected_result = (
        value * a**a_scale_exponent * hubble_param**hubble_scale_exponent * 1.0
    )
    assert result == expected_result, f"Expected {expected_result}, but got {result}"


def test_SFTtoAge():
    # Test with some arbitrary scale factor
    a = 0.5

    result = SFTtoAge(a)

    # Check if the result is as expected
    expected_result = cosmo.lookback_time((1 / a) - 1).value
    assert result == expected_result, f"Expected {expected_result}, but got {result}"


def test_read_yaml(tmp_path):
    # Create a temporary YAML file in the temporary directory
    test_file = tmp_path / "test.yaml"

    # Data to write to the YAML file
    data = {"key": "value", "numbers": [1, 2, 3]}

    # Write the YAML data to the file
    with open(test_file, "w") as file:
        yaml.dump(data, file)

    # Use the read_yaml function to read the data back from the file
    result = read_yaml(str(test_file))

    # Check that the data read from the file matches the original data
    assert (
        result == data
    ), "The data read from the YAML file does not match the expected data"


def test_read_yaml_error_handling(tmp_path):
    # Define a path to a non-existent file
    non_existent_file = tmp_path / "non_existent.yaml"

    # Expect a RuntimeError when trying to read a non-existent file
    with pytest.raises(RuntimeError) as excinfo:
        read_yaml(str(non_existent_file))

    assert "Something went wrong while reading yaml file" in str(
        excinfo.value
    ), "Expected RuntimeError not raised"
