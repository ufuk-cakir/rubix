import os
from typing import Callable, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from rubix.galaxy import IllustrisAPI, get_input_handler
from rubix.galaxy.alignment import center_particles
from rubix.logger import get_logger
from rubix.utils import load_galaxy_data, get_config


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
    config = get_config(config)
    # Check if save_name is present in the config

    # Setup a logger based on the config
    logger_config = config["logger"] if "logger" in config else None

    logger = get_logger(logger_config)

    # Get the variables from the config
    output_path = config["data/output_path"]
    save_name = config["data/save_name"]
    if os.path.exists(os.path.join(output_path, f"rubix_galaxy_{save_name}.h5")):
        logger.warning("Rubix galaxy file already exists, skipping conversion")
        return os.path.join(output_path, f"rubix_galaxy_{save_name}.h5")

    # If the simulationtype is IllustrisAPI, get data from IllustrisAPI
    simulation_name = config["data/simulation/name"]

    # TODO: we can do this more elgantly
    if simulation_name == "IllustrisAPI":
        logger.info("Loading data from IllustrisAPI")
        save_data_path = os.path.join(output_path, "illustris_api_data")
        api = IllustrisAPI(
            **config["data/simulation/args"],
            save_data_path=save_data_path,
            logger=logger,
        )

        api.load_galaxy()

        # Load the saved data into the input handler
    logger.info("Loading data into input handler")
    input_handler = get_input_handler(config, logger=logger)
    input_handler.to_rubix(output_path=output_path, save_name=save_name)

    return os.path.join(output_path, f"rubix_galaxy_{save_name}.h5")


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


def prepare_input(config: Union[dict, str]) -> Tuple[
    Float[Array, "n_particles 3"],
    Float[Array, "n_particles 3"],
    Float[Array, " n_particles"],
    Float[Array, " n_particles"],
    Float[Array, " n_particles"],
    float,
]:

    logger_config = config["logger"] if "logger" in config else None  # type:ignore
    logger = get_logger(logger_config)
    file_path = config["data/output_path"]  # type:ignore
    save_name = config["data/save_name"]  # type:ignore
    file_path = os.path.join(file_path, f"rubix_galaxy_{save_name}.h5")

    # Load the data from the file
    # TODO: maybe also pass the units here, currently this is not used
    data, units = load_galaxy_data(file_path)

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

    # Check if we should only use a subset of the data for testing and memory reasons
    if "data" in config:
        if "subset" in config["data"]:  # type:ignore
            if config["data"]["subset"]["use_subset"]:  # type:ignore
                size = config["data"]["subset"]["subset_size"]  # type:ignore
                # Randomly sample indices
                # Set random seed for reproducibility
                np.random.seed(42)
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
        halfmassrad_stars,
    )


def get_rubix_data(config: Union[dict, str]) -> Tuple[
    Float[Array, "n_particles 3"],
    Float[Array, "n_particles 3"],
    Float[Array, " n_particles"],
    Float[Array, " n_particles"],
    Float[Array, " n_particles"],
    float,
]:
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
