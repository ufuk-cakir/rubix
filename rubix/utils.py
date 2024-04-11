# Description: Utility functions for Rubix

import numpy as np
from astropy.cosmology import Planck15 as cosmo


def convert_values_to_physical(
    value,
    a,
    a_scale_exponent,
    hubble_param,
    hubble_scale_exponent,
    CGS_conversion_factor,
):
    """Convert values from cosmological simulations to physical units
    Source: https://kateharborne.github.io/SimSpin/examples/generating_hdf5.html#attributes

    Parameters
    ----------
    value : float
        Value from Simulation Parameter to be converted
    a : float
        Scale factor, given as 1/(1+z)
    a_scale_exponent : float
        Exponent of the scale factor
    hubble_param : float
        Hubble parameter
    hubble_scale_exponent : float
        Exponent of the Hubble parameter
    CGS_conversion_factor : float
        Conversion factor to CGS units

    Returns
    -------
    float
        Value in physical units
    """
    # check if CGS_conversion_factor is 0
    if CGS_conversion_factor == 0:
        # Sometimes IllustrisTNG returns 0 for the conversion factor, in which case we assume it is already in CGS
        CGS_conversion_factor = 1.0
    # convert to physical units
    value = (
        value
        * a**a_scale_exponent
        * hubble_param**hubble_scale_exponent
        * CGS_conversion_factor
    )
    return value


def SFTtoAge(a):
    """Convert scale factor to age in Gyr.

    The lookback time is calculated as the difference between current age
    of the universe and the age at redshift z=1/a - 1.

    This hence gives the age of the star formed at redshift z=1/a - 1.

    """
    # TODO maybe implement this in JAX?
    # TODO CHECK IF THIS IS WHAT WE WANT
    return cosmo.lookback_time((1 / a) - 1).value
