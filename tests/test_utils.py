import pytest  # type: ignore # noqa
from rubix.utils import convert_values_to_physical, SFTtoAge, print_hdf5_file_structure, read_yaml
from astropy.cosmology import Planck15 as cosmo
import h5py


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





def test_hdf5_file_structure(tmp_path):
    # Create a temporary HDF5 file
    hdf5_file = tmp_path / "test_file.h5"
    with h5py.File(hdf5_file, "w") as f:
        group = f.create_group("group")
        dataset = group.create_dataset("dataset", (100,), dtype='i')
        dataset[:] = range(100)

    # Run the function to test
    output = print_hdf5_file_structure(str(hdf5_file))

    # Define expected output, adjusting for additional parentheses and newline
    expected_output = f"File: {str(hdf5_file)}\nGroup: group\n    Dataset: dataset (int32) ((100,))\n"

    # Check if the output matches the expected output
    assert output.strip() == expected_output.strip()

def test_empty_hdf5_file(tmp_path):
    # Test with an empty HDF5 file
    hdf5_file = tmp_path / "empty.h5"
    with h5py.File(hdf5_file, "w") as f:
        pass  # No groups or datasets created

    # Run the function to test
    output = print_hdf5_file_structure(str(hdf5_file))

    # Define expected output, adjusting for the extra newline
    expected_output = f"File: {str(hdf5_file)}\n"

    # Check if the output matches the expected output
    assert output.strip() == expected_output.strip()
    
    
def test_read_yaml_wrong_path():
    # Test with a non-existent YAML file
    with pytest.raises(Exception) as e:
        read_yaml("non_existent_file.yaml")
        