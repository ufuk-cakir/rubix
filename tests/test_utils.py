import pytest  # type: ignore # noqa
from rubix.utils import convert_values_to_physical, SFTtoAge
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
