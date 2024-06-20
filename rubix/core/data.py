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
from rubix import config


def create_dynamic_dataclass(name, fields):
    annotations = {field_name: Optional[jnp.ndarray] for field_name in fields}
    return make_dataclass(name, [(field_name, annotation, field(default=None)) for field_name, annotation in annotations.items()])

Galaxy = create_dynamic_dataclass("Galaxy", config["BaseHandler"]["galaxy"])
StarsData = create_dynamic_dataclass("StarsData", config["BaseHandler"]["particles"]["stars"]) 
GasData = create_dynamic_dataclass("GasData", config["BaseHandler"]["particles"]["gas"])
#gas_data_instance = GasData()
#print(gas_data_instance.__dict__) 

@dataclass
class RubixData:
    """
    This class is used to store the Rubix data in a structured format.
    It is constructed in a dynamic way based on the configuration file.
    """
    galaxy: Optional[Galaxy] = None
    stars: Optional[StarsData] = None
    gas: Optional[GasData] = None

#rubixdata = RubixData(Galaxy(), StarsData(), GasData())
#print(rubixdata)
#print(rubixdata.__dict__)


def convert_to_rubix(config: Union[dict, str]):
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

    logger_config = config["logger"] if "logger" in config else None  # type:ignore
    logger = get_logger(logger_config)
    file_path = config["output_path"]
    file_path = os.path.join(file_path, "rubix_galaxy.h5")

    # Load the data from the file
    data, units = load_galaxy_data(file_path)

    file_path = config["output_path"]  # type:ignore
    file_path = os.path.join(file_path, "rubix_galaxy.h5")

    # Load the data from the file
    # TODO: maybe also pass the units here, currently this is not used
    data, units = load_galaxy_data(file_path)

    rubixdata = RubixData(Galaxy(), StarsData(), GasData())
    print(rubixdata)
    print(rubixdata.__dict__)
    print(rubixdata.galaxy)
    

    if "stars" in config["data"]["args"]["particle_type"]:
        stellar_coordinates = data["particle_data"]["stars"]["coords"]
        stellar_velocities = data["particle_data"]["stars"]["velocity"]
        galaxy_center = data["subhalo_center"]
        halfmassrad_stars = data["subhalo_halfmassrad_stars"]

        # Center the particles
        new_stellar_coordinates, new_stellar_velocities = center_particles(
            stellar_coordinates, stellar_velocities, galaxy_center
        )

        # Load the metallicity and age data

        stars_metallicity = data["particle_data"]["stars"]["metallicity"]
        stars_mass = data["particle_data"]["stars"]["mass"]
        stars_age = data["particle_data"]["stars"]["age"]

    if "gas" in config["data"]["args"]["particle_type"]:
        gas_coordinates = data["particle_data"]["gas"]["coords"]
        gas_velocities = data["particle_data"]["gas"]["velocity"]
        galaxy_center = data["subhalo_center"]
        halfmassrad_stars = data["subhalo_halfmassrad_stars"]

        # Center the particles
        new_gas_coordinates, new_gas_velocities = center_particles(
            gas_coordinates, gas_velocities, galaxy_center
        )

        # Load the metallicity and age data

        gas_metallicity = data["particle_data"]["gas"]["metallicity"]
        gas_density = data["particle_data"]["gas"]["density"]
        gas_mass = data["particle_data"]["gas"]["mass"]
        gas_sfr = data["particle_data"]["gas"]["sfr"]
        gas_internal_energy = data["particle_data"]["gas"]["internal_energy"]
        gas_electron_abundance = data["particle_data"]["gas"]["electron_abundance"]

    # Check if we should only use a subset of the data for testing and memory reasons
    if "data" in config:
        if "subset" in config["data"]:  # type:ignore
            if config["data"]["subset"]["use_subset"]:  # type:ignore
                size = config["data"]["subset"]["subset_size"]  # type:ignore
                # Randomly sample indices
                # Set random seed for reproducibility
                np.random.seed(42)
                if "stars" in config["data"]["args"]["particle_type"]:
                    indices = np.random.choice(
                        np.arange(new_stellar_coordinates.shape[0]),
                        size=size,  # type:ignore
                        replace=False,
                    )  # type:ignore

                    new_stellar_coordinates = new_stellar_coordinates[indices]
                    new_stellar_velocities = new_stellar_velocities[indices]
                    stars_metallicity = stars_metallicity[indices]
                    stars_mass = stars_mass[indices]
                    stars_age = stars_age[indices]
                    logger.warning(
                        f"The Subset value is set in config. Using only subset of size {size}"
                    )
                if "gas" in config["data"]["args"]["particle_type"]:
                    indices = np.random.choice(
                        np.arange(new_gas_coordinates.shape[0]),
                        size=size,  # type:ignore
                        replace=False,
                    )
                    new_gas_coordinates = new_gas_coordinates[indices]
                    new_gas_velocities = new_gas_velocities[indices]
                    gas_metallicity = gas_metallicity[indices]
                    gas_mass = gas_mass[indices]
                    gas_density = gas_density[indices]
                    gas_sfr = gas_sfr[indices]
                    gas_internal_energy = gas_internal_energy[indices]
                    gas_electron_abundance = gas_electron_abundance[indices]

    if "stars" in config["data"]["args"]["particle_type"] and "gas" in config["data"]["args"]["particle_type"]:
        return (
            new_stellar_coordinates,
            new_stellar_velocities,
            stars_metallicity,
            stars_mass,
            stars_age,
            halfmassrad_stars,
            new_gas_coordinates,
            new_gas_velocities,
            gas_metallicity,
            gas_mass,
            gas_density,
            gas_sfr,
            gas_internal_energy,
            gas_electron_abundance,
        )
    elif "stars" in config["data"]["args"]["particle_type"]:
        return (
            new_stellar_coordinates,
            new_stellar_velocities,
            stars_metallicity,
            stars_mass,
            stars_age,
            halfmassrad_stars,
        )
    elif "gas" in config["data"]["args"]["particle_type"]:
        return (
            new_gas_coordinates,
            new_gas_velocities,
            gas_metallicity,
            gas_mass,
            gas_density,
            gas_sfr,
            gas_internal_energy,
            gas_electron_abundance,
            halfmassrad_stars,
        )

