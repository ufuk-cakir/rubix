import time
from typing import Union

import jax
import jax.numpy as jnp
import numpy as np
import sys

from rubix.logger import get_logger
from rubix.pipeline import linear_pipeline as pipeline
from rubix.pipeline import transformer as transformer
from rubix.utils import get_config, get_pipeline_config

from .data import get_reshape_data, get_rubix_data
from .ifu import (
    get_calculate_spectra,
    get_doppler_shift_and_resampling,
    get_scale_spectrum_by_mass,
    get_calculate_datacube,
)
from .rotation import get_galaxy_rotation
from .ssp import get_ssp
from .telescope import get_spaxel_assignment, get_telescope, get_filter_particles
from .psf import get_convolve_psf
from .lsf import get_convolve_lsf
from .noise import get_apply_noise

class RubixPipeline:
    """
    RubixPipeline is responsible for setting up and running the data processing pipeline.

    Parameters
    ----------
    user_config : dict or str
        User configuration for the pipeline.

    Attributes
    ----------
    user_config : dict
        Parsed user configuration.
    pipeline_config : dict
        Configuration for the pipeline.
    logger : Logger
        Logger instance for logging messages.
    ssp : object
        Stellar population synthesis model.
    telescope : object
        Telescope configuration.
    data : dict
        Dictionary containing particle data.
    func : callable
        Compiled pipeline function to process data.

    Examples
    --------
    >>> from rubix.core.pipeline import RubixPipeline
    >>> config = "path/to/config.yml"
    >>> pipeline = RubixPipeline(config)
    >>> output = pipeline.run()
    >>> ssp_model = pipeline.ssp
    >>> telescope = pipeline.telescope
    """

    def __init__(self, user_config: Union[dict, str]):
        self.user_config = get_config(user_config)
        self.pipeline_config = get_pipeline_config(self.user_config["pipeline"]["name"])
        self.logger = get_logger(self.user_config["logger"])
        self.ssp = get_ssp(self.user_config)
        self.telescope = get_telescope(self.user_config)
        self.data = self._prepare_data()
        self.func = None

    def _prepare_data(self):
        """
        Prepares and loads the data for the pipeline.

        Returns
        -------
        dict
            Dictionary containing particle data with keys:
            'n_particles', 'coords', 'velocities', 'metallicity', 'mass', and 'age'.
        """
        # Get the data
        self.logger.info("Getting rubix data...")
        rubixdata = get_rubix_data(
            self.user_config
        )
        star_count = len(rubixdata.stars.coords) if rubixdata.stars.coords is not None else 0
        gas_count = len(rubixdata.gas.coords) if rubixdata.gas.coords is not None else 0
        self.logger.info(f"Data loaded with {star_count} star particles and {gas_count} gas particles.")
        self.logger.info(f"Data loaded with {sys.getsizeof(rubixdata)} properties.")
        # Setup the data dictionary
        # TODO: This is a temporary solution, we need to figure out a better way to handle the data
        # This works, because JAX can trace through the data dictionary
        # Other option may be named tuples or data classes to have fixed keys

        self.logger.debug("Data: %s", rubixdata)
        #self.logger.debug(
        #    "Data Shape: %s",
        #    {k: v.shape for k, v in rubixdata.items() if hasattr(v, "shape")},
        #)

        """
        attributes_star = [attr for attr in dir(rubixdata.stars) if not attr.startswith('__')]
        for attr in attributes_star:
            attribute_value = rubixdata.stars.__getattribute__(attr)  # Get the current value
            jax_array_value = jnp.array(attribute_value)  # Convert it to a JAX array
            setattr(rubixdata.stars, attr, jax_array_value)  # Set the converted value back

        attributes_gas = [attr for attr in dir(rubixdata.gas) if not attr.startswith('__')]
        for attr in attributes_gas:
            attribute_value = rubixdata.gas.__getattribute__(attr)
            jax_array_value = jnp.array(attribute_value)
            setattr(rubixdata.gas, attr, jax_array_value)

        attributes_galaxy = [attr for attr in dir(rubixdata.galaxy) if not attr.startswith('__')]
        for attr in attributes_galaxy:
            attribute_value = rubixdata.galaxy.__getattribute__(attr)
            jax_array_value = jnp.array(attribute_value)
            setattr(rubixdata.galaxy, attr, jax_array_value)
        """
        
        return rubixdata

    def _get_pipeline_functions(self) -> list:
        """
        Sets up the pipeline functions.

        Returns
        -------
        list
            List of functions to be used in the pipeline.
        """
        self.logger.info("Setting up the pipeline...")
        self.logger.debug("Pipeline Configuration: %s", self.pipeline_config)

        # TODO: maybe there is a nicer way to load the functions from the yaml config?
        rotate_galaxy = get_galaxy_rotation(self.user_config)
        filter_particles = get_filter_particles(self.user_config)
        spaxel_assignment = get_spaxel_assignment(self.user_config)
        calculate_spectra = get_calculate_spectra(self.user_config)
        reshape_data = get_reshape_data(self.user_config)
        scale_spectrum_by_mass = get_scale_spectrum_by_mass(self.user_config)
        doppler_shift_and_resampling = get_doppler_shift_and_resampling(
            self.user_config
        )
        calculate_datacube = get_calculate_datacube(self.user_config)
        convolve_psf = get_convolve_psf(self.user_config)
        convolve_lsf = get_convolve_lsf(self.user_config)
        apply_noise = get_apply_noise(self.user_config)

        functions = [
            rotate_galaxy,
            filter_particles,
            spaxel_assignment,
            calculate_spectra,
            reshape_data,
            scale_spectrum_by_mass,
            doppler_shift_and_resampling,
            calculate_datacube,
            convolve_psf,
            convolve_lsf,
            apply_noise,
        ]

        return functions

    # TODO: currently returns dict, but later should return only the IFU cube
    def run(self):
        """
        Runs the data processing pipeline.

        Returns
        -------
        dict
            Output of the pipeline after processing the input data.
        """
        # Create the pipeline
        time_start = time.time()
        functions = self._get_pipeline_functions()
        self._pipeline = pipeline.LinearTransformerPipeline(
            self.pipeline_config, functions
        )

        # Assembling the pipeline
        self.logger.info("Assembling the pipeline...")
        self._pipeline.assemble()

        # Compiling the expressions
        self.logger.info("Compiling the expressions...")
        self.func = self._pipeline.compile_expression()

        # Running the pipeline
        self.logger.info("Running the pipeline on the input data...")
        output = self.func(self.data)

        jax.block_until_ready(output)
        time_end = time.time()

        self.logger.info(
            "Pipeline run completed in %.2f seconds.", time_end - time_start
        )
        return output

    # TODO: implement gradient calculation
    def gradient(self):
        raise NotImplementedError("Gradient calculation is not implemented yet")
