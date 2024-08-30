import os
from typing import Callable, Tuple, Union, Optional
from dataclasses import dataclass, field, make_dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from rubix.galaxy import IllustrisAPI, get_input_handler
from rubix.galaxy.alignment import center_particles
from rubix.logger import get_logger
from rubix.utils import load_galaxy_data, read_yaml
from rubix import config as rubix_config


# Registering the dataclass with JAX for automatic tree traversal
@partial(jax.tree_util.register_pytree_node_class)
@dataclass
class Galaxy:
    redshift: Optional[jnp.ndarray] = None
    center: Optional[jnp.ndarray] = None
    halfmassrad_stars: Optional[jnp.ndarray] = None

    def tree_flatten(self):
        children = (self.redshift, self.center, self.halfmassrad_stars)
        aux_data = {}
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


@partial(jax.tree_util.register_pytree_node_class)
@dataclass
class StarsData:
    # Assuming attributes for StarsData, replace with actual attributes
    coords: Optional[jnp.ndarray] = None
    velocity: Optional[jnp.ndarray] = None
    mass: Optional[jnp.ndarray] = None
    metallicity: Optional[jnp.ndarray] = None
    age: Optional[jnp.ndarray] = None
    pixel_assignment: Optional[jnp.ndarray] = None
    spatial_bin_edges: Optional[jnp.ndarray] = None
    mask: Optional[jnp.ndarray] = None
    spectra: Optional[jnp.ndarray] = None
    datacube: Optional[jnp.ndarray] = None

    def tree_flatten(self):
        children = (
            self.coords,
            self.velocity,
            self.mass,
            self.metallicity,
            self.age,
            self.pixel_assignment,
            self.spatial_bin_edges,
            self.mask,
            self.spectra,
            self.datacube,
        )
        aux_data = {}
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


@partial(jax.tree_util.register_pytree_node_class)
@dataclass
class GasData:
    # Assuming attributes for GasData, replace with actual attributes
    coords: Optional[jnp.ndarray] = None
    velocity: Optional[jnp.ndarray] = None
    mass: Optional[jnp.ndarray] = None
    density: Optional[jnp.ndarray] = None
    internal_energy: Optional[jnp.ndarray] = None
    metallicity: Optional[jnp.ndarray] = None
    sfr: Optional[jnp.ndarray] = None
    electron_abundance: Optional[jnp.ndarray] = None
    pixel_assignment: Optional[jnp.ndarray] = None
    spatial_bin_edges: Optional[jnp.ndarray] = None
    mask: Optional[jnp.ndarray] = None
    spectra: Optional[jnp.ndarray] = None
    datacube: Optional[jnp.ndarray] = None

    def tree_flatten(self):
        children = (
            self.coords,
            self.velocity,
            self.mass,
            self.density,
            self.internal_energy,
            self.metallicity,
            self.sfr,
            self.electron_abundance,
            self.pixel_assignment,
            self.spatial_bin_edges,
            self.mask,
            self.spectra,
            self.datacube,
        )
        aux_data = {}
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


@partial(jax.tree_util.register_pytree_node_class)
@dataclass
class RubixData:
    galaxy: Optional[Galaxy] = None
    stars: Optional[StarsData] = None
    gas: Optional[GasData] = None

    def tree_flatten(self):
        children = (self.galaxy, self.stars, self.gas)
        aux_data = {}
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


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


def prepare_input(config: Union[dict, str]) -> object:
    print(config)

    logger_config = config["logger"] if "logger" in config else None  # type:ignore
    logger = get_logger(logger_config)
    file_path = config["output_path"]
    file_path = os.path.join(file_path, "rubix_galaxy.h5")

    # Load the data from the file
    data, units = load_galaxy_data(file_path)

    file_path = config["output_path"]  # type:ignore
    file_path = os.path.join(file_path, "rubix_galaxy.h5")

    # Galaxy = create_dynamic_dataclass("Galaxy", rubix_config["BaseHandler"]["galaxy"])
    # StarsData = create_dynamic_dataclass("StarsData", rubix_config["BaseHandler"]["particles"]["stars"])
    # GasData = create_dynamic_dataclass("GasData", rubix_config["BaseHandler"]["particles"]["gas"])

    # Load the data from the file
    # TODO: maybe also pass the units here, currently this is not used
    data, units = load_galaxy_data(file_path)

    rubixdata = RubixData(Galaxy(), StarsData(), GasData())

    rubixdata.galaxy.redshift = data["redshift"]
    rubixdata.galaxy.center = data["subhalo_center"]
    rubixdata.galaxy.halfmassrad_stars = data["subhalo_halfmassrad_stars"]

    for partType in config["data"]["args"]["particle_type"]:
        if partType in data["particle_data"]:
            # Convert attributes to JAX arrays and set them on rubixdata
            for attribute, value in data["particle_data"][partType].items():
                jax_value = jnp.array(value)
                setattr(getattr(rubixdata, partType), attribute, jax_value)

            # Center the particles
            logger.info(f"Centering {partType} particles")
            rubixdata = center_particles(rubixdata, partType)

            # Subset the attributes
            for attribute in data["particle_data"][partType].keys():
                attr_value = getattr(getattr(rubixdata, partType), attribute)
                if attr_value.ndim == 2:  # For attributes with shape (N, 3)
                    setattr(
                        getattr(rubixdata, partType),
                        attribute,
                        attr_value[jax_indices, :],
                    )
                else:  # For attributes with shape (N,)
                    setattr(
                        getattr(rubixdata, partType), attribute, attr_value[jax_indices]
                    )

            # Log the subset warning
            logger.warning(
                f"The Subset value is set in config. Using only subset of size {size} for {partType}"
            )

    """
    if "stars" in config["data"]["args"]["particle_type"]:
        for attribute, value in data["particle_data"]["stars"].items():
            jax_value = jnp.array(value)
            setattr(rubixdata.stars, attribute, jax_value)
        rubixdata = center_particles(rubixdata, "stars")

    if "gas" in config["data"]["args"]["particle_type"]:
        for attribute, value in data["particle_data"]["gas"].items():
            jax_value = jnp.array(value)
            setattr(rubixdata.gas, attribute, value)
        rubixdata = center_particles(rubixdata, "gas")

    if "data" in config:
        if "subset" in config["data"]:  # type:ignore
            if config["data"]["subset"]["use_subset"]:  # type:ignore
                size = config["data"]["subset"]["subset_size"]  # type:ignore
                # Randomly sample indices
                # Set random seed for reproducibility
                np.random.seed(42)
                if "stars" in config["data"]["args"]["particle_type"]:
                    indices = np.random.choice(
                    np.arange(len(rubixdata.stars.coords)),
                    size=size,  # type:ignore
                    replace=False,
                )  # type:ignore
                    rubixdata.stars.coords = rubixdata.stars.coords[indices,:]
                    rubixdata.stars.velocity = rubixdata.stars.velocity[indices,:]
                    rubixdata.stars.metallicity = rubixdata.stars.metallicity[indices]
                    rubixdata.stars.mass = rubixdata.stars.mass[indices]
                    rubixdata.stars.age = rubixdata.stars.age[indices]
                    logger.warning(
                    f"The Subset value is set in config. Using only subset of size {size} for stars"
                )
                if "gas" in config["data"]["args"]["particle_type"]:
                    indices = np.random.choice(
                    np.arange(len(rubixdata.stars.coords)),
                    size=size,  # type:ignore
                    replace=False,
                )  # type:ignore
                    rubixdata.gas.coords = rubixdata.gas.coords[indices,:]
                    rubixdata.gas.velocity = rubixdata.gas.velocity[indices,:]
                    rubixdata.gas.metallicity = rubixdata.gas.metallicity[indices]
                    rubixdata.gas.mass = rubixdata.gas.mass[indices]
                    rubixdata.gas.density = rubixdata.gas.density[indices]
                    rubixdata.gas.internal_energy = rubixdata.gas.internal_energy[indices]
                    rubixdata.gas.metallcity = rubixdata.gas.metallicity[indices]
                    rubixdata.gas.sfr = rubixdata.gas.sfr[indices]
                    rubixdata.gas.electron_abundance = rubixdata.gas.electron_abundance[indices]
                    logger.warning(
                    f"The Subset value is set in config. Using only subset of size {size} for gas"
                )
    """

    return rubixdata


def get_rubix_data(config: Union[dict, str]) -> object:
    """Returns the Rubix data

    First converts the data to Rubix format and then prepares the input data.


    """
    convert_to_rubix(config)
    return prepare_input(config)


def get_reshape_data(config: Union[dict, str]) -> Callable:
    """Returns a function to reshape the data

    Maps the `reshape_array` function to the input data dictionary.
    """
    """
    def reshape_data(
        input_data: dict,
        keys=["coords", "velocity", "metallicity", "mass", "age", "pixel_assignment"],
    ) -> dict:
        # TODO:Maybe write this more elegantly
        for key in keys:
            input_data[key] = reshape_array(input_data[key])

        return input_data
    """

    def reshape_data(rubixdata: object) -> object:
        # Check if input_data has 'stars' and 'gas' attributes and process them separately
        if rubixdata.stars.velocity is not None:
            attributes = [
                attr for attr in dir(rubixdata.stars) if not attr.startswith("__")
            ]
            for key in attributes:
                # Get the attribute value; continue to next key if it's None
                attr_value = getattr(rubixdata.stars, key)
                if attr_value is None or not isinstance(
                    attr_value, (jnp.ndarray, np.ndarray)
                ):
                    continue
                # Ensure reshape_array is compatible with JAX arrays
                reshaped_value = reshape_array(attr_value)
                setattr(rubixdata.stars, key, reshaped_value)

        if rubixdata.gas.velocity is not None:
            attributes = [
                attr for attr in dir(rubixdata.gas) if not attr.startswith("__")
            ]
            for key in attributes:
                # Get the attribute value; continue to next key if it's None
                attr_value = getattr(rubixdata.gas, key)
                if attr_value is None or not isinstance(
                    attr_value, (jnp.ndarray, np.ndarray)
                ):
                    continue
                # Ensure reshape_array is compatible with JAX arrays
                reshaped_value = reshape_array(attr_value)
                setattr(rubixdata.gas, key, reshaped_value)

        return rubixdata

    """
    def reshape_data(
        rubixdata: object,
        keys=["coords", "velocity", "metallicity", "mass", "age", "pixel_assignment"],
        ) -> object:
            # TODO:Maybe write this more elegantly
            for key in keys:
                attr_value = getattr(rubixdata.stars, key)
                reshaped_value = reshape_array(attr_value)
                setattr(rubixdata.stars, key, reshaped_value)
            return rubixdata
     """
    return reshape_data
