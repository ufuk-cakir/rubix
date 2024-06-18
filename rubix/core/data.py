import os
from typing import Callable, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from rubix.galaxy import IllustrisAPI, get_input_handler
from rubix.galaxy.alignment import center_particles
from rubix.logger import get_logger
from rubix.utils import load_galaxy_data, read_yaml


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

    file_path = config["output_path"]
    file_path = os.path.join(file_path, "rubix_galaxy.h5")

    # Load the data from the file
    data, units = load_galaxy_data(file_path)
    galaxy_center = data["subhalo_center"]

    # Initialize return variables
    results = []

    # Check for particle types to process
    particle_types = config.get("particle_type", "both")  # Default to both if not specified

    if particle_types in ["both", "stars"]:
        # Process star data
        stellar_coordinates = data["particle_data"]["stars"]["coords"]
        stellar_velocities = data["particle_data"]["stars"]["velocity"]
        new_stellar_coordinates, new_stellar_velocities = center_particles(
            stellar_coordinates, stellar_velocities, galaxy_center
        )
        stars_metallicity = data["particle_data"]["stars"]["metallicity"]
        stars_mass = data["particle_data"]["stars"]["mass"]
        stars_age = data["particle_data"]["stars"]["age"]
        results.extend([
            new_stellar_coordinates, new_stellar_velocities,
            stars_metallicity, stars_mass, stars_age
        ])

    if particle_types in ["both", "gas"]:
        # Process gas data
        gas_coordinates = data["particle_data"]["gas"]["coords"]
        gas_velocities = data["particle_data"]["gas"]["velocity"]
        new_gas_coordinates, new_gas_velocities = center_particles(
            gas_coordinates, gas_velocities, galaxy_center
        )
        gas_metallicity = data["particle_data"]["gas"]["metallicity"]
        gas_mass = data["particle_data"]["gas"]["mass"]
        gas_density = data["particle_data"]["gas"]["density"]
        gas_hsml = data["particle_data"]["gas"]["hsml"]
        gas_sfr = data["particle_data"]["gas"]["sfr"]
        gas_internal_energy = data["particle_data"]["gas"]["internal_energy"]
        gas_electron_abundance = data["particle_data"]["gas"]["electron_abundance"]
        gas_metals = data["particle_data"]["gas"]["metals"]
        results.extend([
            new_gas_coordinates, new_gas_velocities,
            gas_metallicity, gas_mass, gas_density, gas_hsml,
            gas_sfr, gas_internal_energy, gas_electron_abundance, gas_metals
        ])

    return tuple(results)