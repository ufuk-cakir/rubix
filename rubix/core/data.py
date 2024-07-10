import os
from typing import Callable, Tuple, Union, Optional
from dataclasses import dataclass, field, make_dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from rubix.galaxy import IllustrisAPI, get_input_handler
from rubix.galaxy.alignment import center_particles
from rubix.logger import get_logger
from rubix.utils import load_galaxy_data, read_yaml
from rubix import config as rubix_config


class Particles:
    """
    Mixin class to handle subsetting of dataclasses

    Methods:
        apply_subset(indices): Applies subsetting to all fields of the dataclass.
            Each field is updated to only contain elements at the specified indices.
            It gets a random subset of data for speed reason and testing.

    Example usage:
        @dataclass
        class MyData(SubsetMixin):
            field1: List[int]
            field2: List[str]

        data = MyData(field1=[1, 2, 3], field2=['a', 'b', 'c'])
        data.apply_subset([0, 2])  # data now contains elements at indices 0 and 2
    """
    def apply_subset(self, indices):
        """
        Applies subsetting to all fields of the dataclass.

        Parameters:
            indices: List[int]
                The indices to keep in each field of the dataclass
        """
        for field_name, value in self.__dataclass_fields__.items():
            current_value = getattr(self, field_name)
            if current_value is not None:
                setattr(self, field_name, current_value[indices])


def create_dynamic_dataclass(name, fields):
    """
    Create a dataclass dynamically based on the provided fields, all of which are optional and default to None.
    Each field is of type Optional[jnp.ndarray], that it can hold a JAX numpy array or None.
    It inherits from SubsetMixin to allow subsetting of the dataclass using apply_subset method

    Parameters:
        name: str
            The name of the dataclass
        fields: List[str]
            The names of the fields to include in the dataclass

    Returns:
        type
            The dynamically created dataclass
    
    Example usage:
        MyDynamicData = create_dynamic_dataclass('MyDynamicData', ['field1', 'field2'])
        instance = MyDynamicData()
        instance.field1 = jnp.array([1, 2, 3])
        instance.apply_subset([0, 2])  # Applies subsetting to all fields
    """
    annotations = {field_name: Optional[jnp.ndarray] for field_name in fields}
    # Include SubsetMixin in the bases
    return make_dataclass(name, [(field_name, annotation, field(default=None)) for field_name, annotation in annotations.items()], bases=(SubsetMixin,))

@dataclass
class Galaxy:
    # Galaxy class definition here
    pass

@dataclass
class StarsData:
    # StarsData class definition here
    pass

@dataclass
class GasData:
    # GasData class definition here
    pass

@dataclass
class RubixData:
    """
    This class is used to store the Rubix data in a structured format.
    It is constructed in a dynamic way based on the configuration file.

    galaxy:
        Contains general information about the galaxy
        redshift, galaxy center, halfmassrad
    stars:
        Contains information about the stars
    gas:
        Contains information about the gas
    """
    def __init__(self, galaxy: Optional[Galaxy] = None, stars: Optional[StarsData] = None, gas: Optional[GasData] = None):
        self.galaxy = galaxy
        self.stars = stars
        self.gas = gas

    galaxy: Optional[Galaxy] = None
    stars: Optional[StarsData] = None
    gas: Optional[GasData] = None


def convert_to_rubix(config: Union[dict, str]):
    """Converts the data to Rubix format

    This function converts the data to Rubix format. The data can be loaded from an API or from a file, is then
    converted to Rubix format and saved to a file. If the file already exists, the conversion is skipped.

    Parameters
    ----------
    config: dict or str
        The configuration for the conversion. This can be a dictionary or a path to a YAML file containing the configuration.

    Returns
    -------
    str: The path to the output file
    """
    # Check if the file already exists
    # Create the input handler based on the config and create rubix galaxy data
    if isinstance(config, str):
        config = read_yaml(config)

    # Setup a logger based on the config
    logger_config = config["logger"] if "logger" in config else None

    logger = get_logger(logger_config)

    if os.path.exists(os.path.join(config["output_path"], "rubix_galaxy.h5")):
        logger.info("Rubix galaxy file already exists, skipping conversion")
        return config["output_path"]

    # If the simulationtype is IllustrisAPI, get data from IllustrisAPI

    # TODO: we can do this more elgantly
    if "data" in config:
        if config["data"]["name"] == "IllustrisAPI":
            logger.info("Loading data from IllustrisAPI")
            api = IllustrisAPI(**config["data"]["args"], logger=logger)
            api.load_galaxy(**config["data"]["load_galaxy_args"])

            # Load the saved data into the input handler
    logger.info("Loading data into input handler")
    input_handler = get_input_handler(config, logger=logger)
    input_handler.to_rubix(output_path=config["output_path"])

    return config["output_path"]


def reshape_array(
    arr: Float[Array, "n_particles n_features"]
) -> Float[Array, "n_gpus particles_per_gpu n_features"]:
    """Reshapes an array to be compatible with JAX parallelization

    The function reshapes an array of shape (n_particles, n_features) to an array of shape (n_gpus, particles_per_gpu, n_features).

    Padding with zero is added if necessary to ensure that the number of particles per GPU is the same for all GPUs.

    Parameters
    ----------
    arr: jnp.ndarray
        The array to reshape

    Returns
    -------
    jnp.ndarray
        The reshaped array
    """
    n_gpus = jax.device_count()
    n_particles = arr.shape[0]

    # Check if arr is 1D or 2D
    is_1d = arr.ndim == 1

    if is_1d:
        # Convert 1D array to 2D by adding a second dimension
        arr = arr[:, None]
    # Calculate the number of particles per GPU
    particles_per_gpu = (n_particles + n_gpus - 1) // n_gpus

    # Calculate the total number of particles after padding
    total_particles = particles_per_gpu * n_gpus

    # Pad the array with zeros if necessary
    if total_particles > n_particles:
        padding = total_particles - n_particles
        arr = jnp.pad(arr, ((0, padding), (0, 0)), "constant")

    # Reshape the array to (n_gpus, particles_per_gpu, arr.shape[1])
    reshaped_arr = arr.reshape(n_gpus, particles_per_gpu, *arr.shape[1:])

    if is_1d:
        # Remove the second dimension added for 1D case
        reshaped_arr = reshaped_arr.squeeze(-1)
    return reshaped_arr


def prepare_input(config: Union[dict, str]):
    print(config)

    logger_config = config["logger"] if "logger" in config else None  # type:ignore
    logger = get_logger(logger_config)
    file_path = config["output_path"]
    file_path = os.path.join(file_path, "rubix_galaxy.h5")

    # Load the data from the file
    data, units = load_galaxy_data(file_path)

    file_path = config["output_path"]  # type:ignore
    file_path = os.path.join(file_path, "rubix_galaxy.h5")

    Galaxy = create_dynamic_dataclass("Galaxy", rubix_config["BaseHandler"]["galaxy"])
    StarsData = create_dynamic_dataclass("StarsData", rubix_config["BaseHandler"]["particles"]["stars"]) 
    GasData = create_dynamic_dataclass("GasData", rubix_config["BaseHandler"]["particles"]["gas"])

    # Load the data from the file
    # TODO: maybe also pass the units here, currently this is not used
    data, units = load_galaxy_data(file_path)

    rubixdata = RubixData(Galaxy(), StarsData(), GasData())
    
    rubixdata.galaxy.redshift = data["redshift"]
    rubixdata.galaxy.center = data["subhalo_center"]
    rubixdata.galaxy.halfmassrad_stars = data["subhalo_halfmassrad_stars"]

    if "stars" in config["data"]["args"]["particle_type"]:
        for attribute, value in data["particle_data"]["stars"].items():
            setattr(rubixdata.stars, attribute, value)
        rubixdata = center_particles(rubixdata, "stars")

    if "gas" in config["data"]["args"]["particle_type"]:
        for attribute, value in data["particle_data"]["gas"].items():
            setattr(rubixdata.gas, attribute, value)
        rubixdata = center_particles(rubixdata, "gas")

    # Subset handling for both stars and gas if applicable
    if "subset" in config.get("data", {}):
        subset_config = config["data"]["subset"]
        if subset_config.get("use_subset", False):
            size = subset_config["subset_size"]
            np.random.seed(42)  # For reproducibility
            if "stars" in config["data"]["args"]["particle_type"]:
                indices = np.random.choice(len(rubixdata.stars.coords), size=size, replace=False)
                rubixdata.stars.apply_subset(indices)
                logger.warning(f"Using only subset of stars data of size {size}")
            if "gas" in config["data"]["args"]["particle_type"]:
                indices = np.random.choice(len(rubixdata.gas.coords), size=size, replace=False)
                rubixdata.gas.apply_subset(indices)
                logger.warning(f"Using only subset of gas data of size {size}")
    

    return rubixdata


def get_rubix_data(config: Union[dict, str]) -> RubixData:
    """Returns the Rubix data

    First converts the data to Rubix format and then prepares the input data.


    """
    convert_to_rubix(config)
    return prepare_input(config)


def get_reshape_data(config: Union[dict, str]) -> Callable:
    """Returns a function to reshape the data

    Maps the `reshape_array` function to the input data dictionary.
    """

    def reshape_data(
        input_data: dict,
        keys=["coords", "velocities", "metallicity", "mass", "age", "pixel_assignment"],
    ) -> dict:
        # TODO:Maybe write this more elegantly
        for key in keys:
            input_data[key] = reshape_array(input_data[key])

        return input_data

    return reshape_data
