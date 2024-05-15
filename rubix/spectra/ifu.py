import jax.numpy as jnp
from rubix import config


def convert_luminoisty_to_flux(luminosity, observation_lum_dist, observation_z, pixel_size):
    '''Convert luminosity to flux in units erg/s/cm^2/Angstrom as observed by the telescope'''
    CONSTANTS = config["constants"]
    CONST = CONSTANTS.get("LSOL_TO_ERG")/CONSTANTS.get("MPC_TO_CM")**2 
    FACTOR = CONST / (4 * jnp.pi * observation_lum_dist**2) / (1 + observation_z) /pixel_size
    spectral_dist = luminosity * FACTOR
    return spectral_dist



def wavelength_doppler_shift(wavelength, velocity):
    """Calculate the Doppler shift of a wavelength due to a velocity.

    Parameters
    ----------
    wavelength : float
        The wavelength in Angstrom.
    velocity : float
        The velocity in km/s.

    Returns
    -------
    float
        The Doppler shifted wavelength in Angstrom.

    """
    # Calculate the Doppler shift of a wavelength due to a velocity
    SPEED_OF_LIGHT = config["constants"]["SPEED_OF_LIGHT"]
    return wavelength * jnp.exp(velocity / SPEED_OF_LIGHT)