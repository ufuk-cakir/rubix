# Description: Utility functions for Rubix
from astropy.cosmology import Planck15 as cosmo
import yaml
import h5py


def read_yaml(path_to_file: str) -> dict:
    """
    read_yaml Read yaml file into dictionary
    
    Args:
        path_to_file (str): path to the file to read
        
    Returns:
        dict: Either the read yaml file in dictionary form, or an empty
            dictionary if an error occured.
    """
    cfg = {}
    try:
        with open(path_to_file, "r") as cfgfile:
            cfg = yaml.safe_load(cfgfile)
    except Exception as e:
        raise RuntimeError(
            f"Something went wrong while reading yaml file {str(path_to_file)}"
        ) from e
    return cfg


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





def print_hdf5_file_structure(file_path):
    return_string = f"File: {file_path}\n"
    with h5py.File(file_path, "r") as f:
        return_string += _print_hdf5_group_structure(f)
    return return_string


def _print_hdf5_group_structure(group, indent=0):
    return_string = ""
    for key in group.keys():
        sub_group = group[key]
        if isinstance(sub_group, h5py.Group):
            return_string += f"{' ' * indent}Group: {key}\n"
            return_string += _print_hdf5_group_structure(sub_group, indent + 4)
        else:
            return_string += f"{' ' * indent}Dataset: {key} ({sub_group.dtype}) ({sub_group.shape})\n"
    return return_string




def load_galaxy_data(path_to_file: str):
    """
    load_galaxy_data Load galaxy data from a file

    Args:
        path_to_file (str): path to the file to load

    Raises:
        RuntimeError: When an error occurs during loading

    Returns:
        dict: Either the loaded galaxy data, or an empty dictionary if an error occured.
    """
    galaxy_data = {}
    units = {}
    try:
        with h5py.File(path_to_file, "r") as f:
            galaxy_data["subhalo_center"] = f["galaxy/center"][()]
            galaxy_data["subhalo_halfmassrad_stars"] = f["galaxy/halfmassrad_stars"][()]
            galaxy_data["redshift"] = f["galaxy/redshift"][()]
            
            units["galaxy"] = {}
            for key in f["galaxy"].keys():
                units["galaxy"][key] = f["galaxy"][key].attrs["unit"]
            # Load the particle data
            galaxy_data["particle_data"] = {}
            for key in f["particles"].keys():
                galaxy_data["particle_data"][key] = {}
                units[key] = {}
                for field in f["particles"][key].keys():
                    galaxy_data["particle_data"][key][field] = f[f"particles/{key}/{field}"][()]
                    units[key][field] = f[f"particles/{key}/{field}"].attrs["unit"]
            
            
    except Exception as e:
        raise RuntimeError(
            f"Something went wrong while loading galaxy data from file {str(path_to_file)}"
        ) from e
    return galaxy_data, units