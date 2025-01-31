import time
from typing import Union

import jax
import jax.numpy as jnp
import sys

from rubix.logger import get_logger
from rubix.pipeline import linear_pipeline as pipeline
from rubix.pipeline import transformer as transformer
from rubix.utils import get_config, get_pipeline_config
from rubix.utils import read_yaml

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

from jaxtyping import Array, Float, jaxtyped
from beartype import beartype as typechecker

from copy import deepcopy


class RubixPipeline:
    """
    RubixPipeline is responsible for setting up and running the data processing pipeline.

    Args:
        user_config (dict or str): Parsed user configuration for the pipeline.
        pipeline_config (dict): Configuration for the pipeline.
        logger(Logger) : Logger instance for logging messages.
        ssp(object) : Stellar population synthesis model.
        telescope(object) : Telescope configuration.
        data (dict): Dictionary containing particle data.
        func (callable): Compiled pipeline function to process data.

    Example
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
        # self.data = self._prepare_data()
        self.func = None

    # def _prepare_data(self):
    #    """
    #    Prepares and loads the data for the pipeline.

    #    Returns:
    #        Dictionary containing particle data with keys:
    #        'n_particles', 'coords', 'velocities', 'metallicity', 'mass', and 'age'.
    #    """
    #    # Get the data
    #    self.logger.info("Getting rubix data...")
    #    rubixdata = get_rubix_data(self.user_config)
    #    star_count = (
    #        len(rubixdata.stars.coords) if rubixdata.stars.coords is not None else 0
    #    )
    #    gas_count = len(rubixdata.gas.coords) if rubixdata.gas.coords is not None else 0
    #    self.logger.info(
    #        f"Data loaded with {star_count} star particles and {gas_count} gas particles."
    #    )
    #    self.logger.info(f"Data loaded with {sys.getsizeof(rubixdata)} properties.")
    #    # Setup the data dictionary
    #    # TODO: This is a temporary solution, we need to figure out a better way to handle the data
    #    # This works, because JAX can trace through the data dictionary
    #    # Other option may be named tuples or data classes to have fixed keys
    #
    #    self.logger.debug("Data: %s", rubixdata)
    #    # self.logger.debug(
    #    #    "Data Shape: %s",
    #    #    {k: v.shape for k, v in rubixdata.items() if hasattr(v, "shape")},
    #    # )
    #
    #    return rubixdata

    @jaxtyped(typechecker=typechecker)
    def _get_pipeline_functions(self) -> list:
        """
        Sets up the pipeline functions.

        Returns:
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

    def run(self, rubixdata):
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
        output = self.func(rubixdata)

        jax.block_until_ready(output)
        time_end = time.time()

        self.logger.info(
            "Pipeline run completed in %.2f seconds.", time_end - time_start
        )
        # output.stars.mass = jnp.squeeze(output.stars.mass, axis=0)
        # output.stars.age = jnp.squeeze(output.stars.age, axis=0)
        # output.stars.metallicity = jnp.squeeze(output.stars.metallicity, axis=0)
        return output

    # TODO: implement gradient calculation
    def gradient(self, rubixdata, targetdata):
        """
        This function will calculate the gradient of the pipeline, but is yet not implemented.
        """
        # _get_pipeline_functions() returns a list of the transformer functions in the correct order
        # transformers_list = self._get_pipeline_functions()

        # read_cfg = read_yaml("../rubix/config/pipeline_config.yml")

        # read_cfg is a dict. We specifically want read_cfg["calc_ifu"], which has "Transformers" inside.
        # pipeline_cfg = read_cfg["calc_gradient"]

        # tp = pipeline.LinearTransformerPipeline(
        #    pipeline_cfg,  # pipeline_cfg == read_cfg["calc_ifu"]
        #    transformers_list,  # The list of function objects from RubixPipeline
        # )

        # compiled_fn = tp.compile_expression()
        # jac_fn = jax.jacrev(compiled_fn)
        # jacobian = jac_fn(rubixdata)
        # stars_gradient = jacobian.stars.datacube

        # return stars_gradient
        return jax.grad(self.loss, argnums=0)(rubixdata, targetdata)

    def loss(self, rubixdata, targetdata):
        """
        Calculate the mean squared error loss.

        Args:
            data (array-like): The predicted data.
            target (array-like): The target data.

        Returns:
            The mean squared error loss.
        """
        output = self.run(rubixdata)
        loss_value = jnp.sum((output.stars.datacube - targetdata.stars.datacube) ** 2)
        return loss_value

    def grad_only_for_age(self, rubixdata, target):
        # 1) Regular gradient w.r.t. entire rubixdata
        full_grad_fn = jax.grad(self.loss, argnums=0)

        g = full_grad_fn(rubixdata, target)

        # Create a copy of g with zeros in all fields except rubixdata.stars.age
        # The idea is:
        #    if a field name is 'age', keep it
        #    else, zero it out
        def mask_grad_tree(g_subtree, d_subtree):
            # This function is called pairwise on (gradient, data)
            # If the data corresponds to the 'stars.age' array, we keep g_subtree.
            # Otherwise, we return 0 (stop gradient).
            # A direct check might be awkward with pytrees, so let's do a quick hack:
            # You can compare shapes or rely on a known structure, etc.

            # Example: If shape matches rubixdata.stars.age shape
            # or if we pass along a "path" variable with jax.tree_map_with_path
            # For simplicity let's rely on shape matching, though it's not perfect:
            if g_subtree.shape == rubixdata.stars.age.shape and jnp.all(
                d_subtree == rubixdata.stars.age
            ):
                return g_subtree
            else:
                return jnp.zeros_like(g_subtree)

        # We assume rubixdata and g share structure
        g_filtered = jax.tree_util.tree_map(mask_grad_tree, g, rubixdata)
        return g_filtered
