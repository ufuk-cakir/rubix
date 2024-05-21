import os
from typing import Callable, Union

import jax
import jax.numpy as jnp
import numpy as np

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


def reshape_array(arr):
    n_gpus = jax.device_count()
    n_particles = arr.shape[0]

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

    return reshaped_arr


def prepare_input(config: Union[dict, str]):

    logger_config = config["logger"] if "logger" in config else None  # type:ignore
    logger = get_logger(logger_config)
    file_path = config["output_path"]  # type:ignore
    file_path = os.path.join(file_path, "rubix_galaxy.h5")

    # Load the data from the file
    # TODO: maybe also pass the units here, currently this is not used
    data, units = load_galaxy_data(file_path)

    stellar_coordinates = data["particle_data"]["stars"]["coords"]
    stellar_velocities = data["particle_data"]["stars"]["velocity"]
    galaxy_center = data["subhalo_center"]

    # Center the particles
    new_stellar_coordinates, new_stellar_velocities = center_particles(
        stellar_coordinates, stellar_velocities, galaxy_center
    )

    # Load the metallicity and age data

    stars_metallicity = data["particle_data"]["stars"]["metallicity"]
    stars_mass = data["particle_data"]["stars"]["mass"]
    stars_age = data["particle_data"]["stars"]["age"]

    # Reshape the arrays

    # new_stellar_coordinates = reshape_array(new_stellar_coordinates)
    # new_stellar_velocities = reshape_array(new_stellar_velocities)
    # stars_metallicity = reshape_array(stars_metallicity)
    # stars_mass = reshape_array(stars_mass)
    # stars_age = reshape_array(stars_age)

    # check if we should only use a subset of the data for testing and memory reasons

    if "data" in config:
        if "subset" in config["data"]:  # type:ignore
            if config["data"]["subset"]["use_subset"]:  # type:ignore
                size = config["data"]["subset"]["subset_size"]  # type:ignore
                # Randomly sample indices
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

    return (
        new_stellar_coordinates,
        new_stellar_velocities,
        stars_metallicity,
        stars_mass,
        stars_age,
    )


def get_rubix_data(config: Union[dict, str]):
    convert_to_rubix(config)
    return prepare_input(config)


def get_reshape_data(config: Union[dict, str]) -> Callable:

    def reshape_data(
        input_data: dict,
        keys=["coords", "velocities", "metallicity", "mass", "age", "pixel_assignment"],
    ) -> dict:
        for key in keys:
            input_data[key] = reshape_array(input_data[key])

        return input_data

    return reshape_data
