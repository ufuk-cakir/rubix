from rubix.galaxy._input_handler._illustris_api import IllustrisAPI
import os


def _download_galaxy(subhalo_id,simulation="TNG50-1", snapshot=99):
    '''This function downloads the galaxy data from the Illustris API.
    
    Returns
    -------
    dict
        The galaxy data.
    '''
    
    # Load the API key
    api_key = _get_api_key()
    
    # Load the galaxy data
    illustris_api = IllustrisAPI(api_key, simulation=simulation, snapshot=snapshot)
    galaxy_data = illustris_api.load_galaxy(id = subhalo_id, verbose = True)
    
    return galaxy_data





def _get_api_key():
    '''This function loads the API key from the environment variable ILLUSTRIS_API_KEY.
    
    Returns
    -------
    str
        The API key.
    '''
    
    key = os.getenv("ILLUSTRIS_API_KEY")
    if key is None:
        raise ValueError("Please set the environment variable ILLUSTRIS_API_KEY.")
    return key

def _convert_values_to_physical(
    value,
    a,
    a_scale_exponent,
    hubble_param,
    hubble_scale_exponent,
    CGS_conversion_factor,
):
    """Convert values to physical units
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

